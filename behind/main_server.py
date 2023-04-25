import torch
from flask import Flask, request, json
from flask_cors import CORS
import logging
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
import base64
import logging as log
import sys, os, cv2
import numpy as np
from argparse import ArgumentParser, SUPPRESS
from torchvision import transforms
import json

from openvino.inference_engine import IENetwork, IECore

app = Flask(__name__)
app.secret_key = b'_5#y3332323xc'
CORS(app, origins='*')

logger = logging.getLogger('Model')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
# 导入num_to_class映射字典
with open('num_to_class.txt', 'r') as f:
    num_to_class = json.loads(f.read())


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DeepModel(metaclass=Singleton):
    pass

    def __init__(self, args):
        self.initVariable(args)

    def initVariable(self, args):
        ie = IECore()

        model_xml = args.model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        net = ie.read_network(model=model_xml, weights=model_bin)

        self.input_blob = next(iter(net.input_info))
        self.exec_net = ie.load_network(network=net, device_name=args.device)

    def pre_process_image(self, image):
        # Normalize to keep data between 0 - 1
        '''
        这一块数据处理要和数据训练是数据的格式弄成一致的，否则训练效果很垃圾。
        '''
        processedImg = (image / 255.0)
        processedImg = torch.from_numpy(processedImg)
        x = processedImg
        x = torch.unsqueeze(x, 0)
        x = x.permute(0, 3, 1, 2)
        x = normalize(x)
        # 将bgr变为rgb
        x = x[:, [2, 1, 0], :, :]

        processedImg = x
        return image, processedImg

    def main_IE_infer(self, image):
        image = cv2.resize(image, (224, 224))
        image, preimg = self.pre_process_image(image)
        outputs = self.exec_net.infer(inputs={self.input_blob: preimg})

        return outputs

    def do_recognize(self, img):
        image = img
        outputs = self.main_IE_infer(image)
        results = outputs["199"]
        # 将 numpy.ndarray 转换为 torch.Tensor
        x_tensor = torch.from_numpy(results)
        # 对 tensor 进行 softmax 操作
        y_tensor = torch.nn.functional.softmax(x_tensor, dim=1)
        # 将 tensor 转换回 numpy.ndarray
        results = y_tensor.detach().numpy()
        label_index = np.ndarray.argmax(results, axis=1)
        prob = results[0][label_index]
        print(label_index, prob)
        label_name = num_to_class[str(label_index[0])]
        print(label_name)
        return label_name + " Probability:" + str(prob[0])


@app.route('/', methods=['GET'])
def index():
    return "works"


@app.route('/recognize', methods=['POST', 'GET'])
def recognize_with_base64string():
    if request.method == 'POST':
        req_data = request.get_json()

        image_base64 = req_data['imageData']
        if image_base64 is None or len(image_base64) < 10:
            response = app.response_class(
                response=json.dumps({"result": {"status": -3, "text": ""}}),
                status=200,
                mimetype='application/json'
            )
            return response
        base64string = base64.b64decode(image_base64)

        img_array = np.frombuffer(base64string, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        rec_result = tager.do_recognize(img)
        response = app.response_class(
            response=rec_result,
            status=200,
            mimetype='application/text'
        )
        return response


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", default='../ir_models/model.xml',
                      help="Required. Path to an .xml file with a trained model", type=str)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU or MYRIAD is "
                           "acceptable. Sample will look for a suitable plugin for device specified. Default value is CPU",
                      default="CPU", type=str)

    return parser


if __name__ == '__main__':
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    tager = DeepModel(args)
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(6767)  # flask端口
    IOLoop.instance().start()
