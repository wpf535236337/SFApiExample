from scene_classification.scene_infer import ImageClassification
from options import parse_args


class ApiExample(object):
    def __init__(self):
        self.args = parse_args()
        self.clas = ImageClassification(self.args)

    def __call__(self, img):
        prob, label = self.clas.predict(img)
        return prob, label


if __name__ == '__main__':
    scene_rec = ApiExample()
    img = 'scene_classification/test_input/brown_bear.png'
    print(scene_rec(img))