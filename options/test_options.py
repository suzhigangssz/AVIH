from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # rewrite devalue values
        # parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        # Privacy protection algorithm parameter setting
        parser.add_argument('--root', help='Input directory with images.', type=str, default='./data/align')  # cifar-10-batches-py|align
        parser.add_argument('--task', help='Target task. [face_recognition | classification]', type=str, default='face_recognition')
        parser.add_argument('--init', help='Initialization method. [random | original]', type=str, default='random')
        parser.add_argument('--num_iter', help='Number of iterations.', type=int, default=800)
        parser.add_argument('--src_model', help='White-box model', type=str, default='ArcFace')
        parser.add_argument('--batch_size', help='batch size', type=int, default=1)
        parser.add_argument('--output', help='output directory', type=str, default='./result/')
        parser.add_argument('--c', help='Weight of variance consistency loss', type=float, default='0.001')

        return parser
