from pathlib import Path
import pytest
from PIL import Image
from coffee_classifier.data import prepare, image_array, CLASSES


def fixture_data(root):
    for split in ['train','test']:
        for label,name in enumerate(CLASSES):
            folder=root/split/name
            folder.mkdir(parents=True)
            for i in range(5):
                Image.new('RGB',(8,8),(label*50,i*20,10 if split=='train' else 200)).save(folder/f'{i}.png')


def test_duplicate_disjoint_and_repeatable(tmp_path):
    fixture_data(tmp_path)
    source=tmp_path/'test/Dark/0.png'
    (tmp_path/'train/Dark/copy.png').write_bytes(source.read_bytes())
    first=prepare(tmp_path)
    assert first==prepare(tmp_path)
    assert len(first['excluded'])==1
    hashes=[r['sha256'] for r in first['rows']]
    assert len(hashes)==len(set(hashes))
    assert first['counts']=={'train':16,'val':4,'test':20}


def test_conflicting_labels_fail(tmp_path):
    fixture_data(tmp_path)
    (tmp_path/'train/Light/conflict.png').write_bytes((tmp_path/'test/Dark/0.png').read_bytes())
    with pytest.raises(ValueError,match='Conflicting labels'):
        prepare(tmp_path)


def test_rgb_preprocessing(tmp_path):
    path=tmp_path/'red.png'
    Image.new('RGB',(10,10),(255,0,0)).save(path)
    array=image_array(path)
    assert array.shape==(50,50,3)
    assert array[0,0].tolist()==[1.0,0.0,0.0]


def test_corrupt_image_fails(tmp_path):
    fixture_data(tmp_path)
    (tmp_path/'train/Dark/broken.png').write_text('broken')
    with pytest.raises(OSError):
        prepare(tmp_path)
