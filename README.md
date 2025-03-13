**# SAM2_streaming**
[**storagy-repo-5**](https://github.com/addinedu-advance-3rd/storagy-repo-5) 리포지토리에 submodule화하기 위해 **forked**된 [**SAM2_streaming**](https://github.com/khw11044/SAM2_streaming) 리포지토리입니다.

아래부터는 2025.03.13(목) 기준 원본 README.md의 내용입니다.


# segment-anything-2 real-time WebCam Streaming
Run Segment Anything Model 2 on a **live video stream**

**# SAM2 live video stream** 
**# SAM2 streaming** 
**# real-time SAM2** 
**# webcam SAM2**


[깃헙링크](https://github.com/khw11044/SAM2_streaming)


## News
- 27/11/2024 : 최초 SAM2 실시간 세그멘테이션 코드 성공 
- sam2, sam2.1 둘다 가능

## Demo

![segmentation](output_gif/segmentation.gif)

</div>


## Getting Started

### Installation

반드시 파이썬 버전은 3.11 이상이어야 합니다

```bash
conda create -n seg python=3.11 -y

conda activate seg 

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

```


```bash
pip install -e .

pip install -e ".[notebooks]"

```

**위에 명령어 후 아래와 같은 오류 발생시**

```bash

        File "/tmp/pip-build-env-v31jxhmj/overlay/lib/python3.11/site-packages/torch/__init__.py", line 367, in <module>
          from torch._C import *  # noqa: F403
          ^^^^^^^^^^^^^^^^^^^^^^
      ImportError: /tmp/pip-build-env-v31jxhmj/overlay/lib/python3.11/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× Getting requirements to build editable did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.

```

다음과 같은 명령어 진행 

```bash
python setup.py build_ext --inplace
```

다음으로 필요 라이브러리 설치 

```bash

pip install -r requirements.txt

```


### Download Checkpoint

SAM2 모델 다운 

```bash
cd checkpoints/sam2

./download_ckpts.sh

cd ../..
```

SAM2.1 모델 다운 

```bash
cd checkpoints/sam2.1

./download_ckpts.sh

cd ../..
```

자 이제 SAM2를 실시간 스트리밍 화면에 적용할 준비가 됐습니다.

### Demo streaming 

1. mp4 파일에서 테스트 해보기 

```python
python demo.py
```

2. webcam에서 실시간 sam2 적용하기 - 마우스로 바운딩 박스를 그려서 sam2할 객체 지정 

```python 
python demo_webcam_box.py
```

![2](https://github.com/user-attachments/assets/0d0ef6b6-6037-4269-ab89-50a4628dccd1)


3. webcam에서 실시간 sam2 적용하기 - 마우스 point 클릭으로 sam2할 객체 지정 

```python 
python demo_webcam_point.py
```

![3_1](https://github.com/user-attachments/assets/5ce081cc-74a7-4765-a63e-461b164537c4)

![3_2](https://github.com/user-attachments/assets/2fab19e2-4e42-442e-84cc-8cf4799f2386)


4. webcam에서 실시간 sam2 적용하기 - 첫 프레임에 사람 tracking 

```python 
python demo_webcam_yolo.py
```

![4](https://github.com/user-attachments/assets/93f5477e-1a0c-48c6-807d-33bdeed06ad6)


## References:

- SAM2 Repository: https://github.com/facebookresearch/segment-anything-2

- https://github.com/Gy920/segment-anything-2-real-time/tree/main