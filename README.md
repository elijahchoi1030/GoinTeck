고인특 프로젝트 준비


1. 가상환경 설정 (기존 github readme 기준)

conda create -n swin python=3.7 -y
conda activate swin

# conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
-> 공식 깃헙에서는 주석처리한 윗 줄로 되어있는데, 제 GPU가 RTX 3070에 11.4가 깔려있어서 그런지 (nvcc --version 기준)
안돌아가서 아래 줄로 환경셋팅 했습니다. 

pip install timm==0.4.12
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8

2. 실행 코드
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --cfg ./configs/swin/swin_tiny_patch4_window7_cifar100_finetune.yaml --batch-size 64  --pretrained swin_base_patch4_window7_224.pth --data-path ./data/cifar_dataset --accumulation-steps 2 --use-checkpoint
-> 아무리 돌려봐도 GPU 문제인 것 같은데, 제 실력으로 해결이 안됩니다....
일단 NCCL이 깔려있지 않아서 생기는 문제 같은데, 윈도우에서 바로 NCCL 하는 법을 못 찾아서...
우분투를 깔아서 해보다가 우분투에서 nvidia driver가 안깔려서 포기했습니다. 

3. 참고 사항
    swin version 2가 출시된 것으로 보입니다. 그러나 그냥 swin으로 가정하고 코드를 짰습니다. 

4. 수정한 사항
    자체적으로 CIFAR을 지원하지는 않는다고 합니다. "data/build.py/line 122 We only support ImageNet Now"
    이를 해결하기 위해 일단 swin_tiny_patch4_window7_cifar100_finetune.yaml 파일을 만들고
    data/build.py/line 120-121 을 작성하였습니다. 
    근데 yaml 파일에 32x32라고 쓴것 만으로 돌아갈지는 잘 모르겠습니다. 

5. 수정이 필요한 사항
models/swin_transformer.py/line 111 :self.softmax = nn.Softmax(dim=-1) 
요기 부분을 바꾸면 될 것 같습니다. 

