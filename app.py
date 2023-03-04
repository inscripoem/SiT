import torch
from torchvision import transforms
from PIL import Image
import vision_transformer as vits
from datasets.load_dataset import pets_dist, large_data_dist
import gradio as gr

student = None
dataset = None

def load_model(ckp_path, img_size=64):
    global student
    if ckp_path is None:
        return '请在左边框选择模型！'
    student = vits.__dict__['vit_small'](img_size=[img_size], num_classes=2)
    checkpoint = torch.load(ckp_path.name)['student']
    msg = student.load_state_dict(checkpoint)
    print('Loaded model with msg: ', msg)
    student.cuda().eval()
    return msg

def load_dataset(dataset_name = 'Pets', dataset_path = '', img_size=64):
    global dataset
    transform = transforms.Compose([      
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    try:
        if dataset_name == 'Pets':
            dataset = pets_dist(dataset_path, 'test', False, '', transform=transform)
        elif dataset_name == 'large_data':
            dataset = large_data_dist(dataset_path, 'test', False, '', transform=transform)
        return '数据集加载成功!'
    except:
        return '数据集加载失败!'


def classify(img, img_size=64):
    global student
    transform = transforms.Compose([      
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    if student is None:
        return '请先加载模型!'
    img = transform(img).unsqueeze(0).cuda()
    output = student(img, classify=True)
    output.cpu().detach().tolist()
    return '阴性(狗): {:.2f} \n阳性(猫): {:.2f} \n 综合判别: {}'.format(output[0][0], output[0][1], '阳性(猫)' if output[0][1] > output[0][0] else '阴性(狗)')

def test(batch_size, progress=gr.Progress()):
    global dataset
    global student
    if student is None or dataset is None:
        return '请先加载模型或数据集!'
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    correct = 0
    total = len(dataset)
    for i, (image, label) in progress.tqdm(enumerate(dataloader)):
        image = image.cuda()
        label = label[0].cuda()
        output = student(image, classify=True)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
    return '\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, total,
        100. * correct / total)

if __name__ == '__main__':
    with gr.Blocks() as app:
        gr.Markdown(
            "# <center> SiT下游图像分类"
        )
        with gr.Row():
            with gr.Column():
                model_path = gr.File(label='模型')
            with gr.Column():
                with gr.Row():
                    img_size = gr.Slider(minimum=32, maximum=512, step=16, value=64, label='图片尺寸')
                    btn = gr.Button(value='加载模型')
                model_msg = gr.Textbox(label='加载输出')
                btn.click(load_model, inputs=[model_path, img_size], outputs=[model_msg])
        with gr.Tabs():
            with gr.TabItem('分类'):   
                with gr.Row():
                    with gr.Column():
                        img = gr.Image(label='图片')
                    with gr.Column():
                        btn2 = gr.Button(value='分类')
                        output = gr.Textbox(label='输出')
                    btn2.click(classify, inputs=[img, img_size], outputs=[output])
            with gr.TabItem('测试'):
                dataset_path = gr.Textbox(label='数据集路径', value='G:\DeepLearning\SiT_docker\dataset\large_data', placeholder='path/to/datadet', interactive=True)
                with gr.Row():
                    with gr.Column():
                        dataset = gr.Dropdown(label='数据集', choices=['Pets', 'large_data'], value='large_data')
                    with gr.Column():
                        batch_size = gr.Slider(minimum=16, maximum=512, step=16, value=1, label='batch size')
                with gr.Row():
                    with gr.Column():
                        dataset_msg = gr.Textbox(label='数据集加载输出')
                    with gr.Column():
                        btn3 = gr.Button(value='开始测试')
                output2 = gr.Textbox(label='输出')
                dataset.change(load_dataset, inputs=[dataset, dataset_path, img_size], outputs=[dataset_msg])
                btn3.click(test, inputs=[batch_size], outputs=[output2])

    app.queue(concurrency_count=1).launch()
