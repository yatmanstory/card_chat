
### 시작하기 (Setup Guide)
이 프로젝트를 로컬 환경에서 실행하기 위한 설정 방법입니다. 
macOS 터미널 기준으로 작성되었습니다.

#### 1. 저장소 복제 (Clone)
먼저 GitHub에 있는 코드를 내 컴퓨터로 가져옵니다.

```Bash
git clone <레포지토리-주소>
cd <폴더-이름>
```

#### 2. 가상환경 생성 (Create Virtual Environment)
다른 프로젝트와의 패키지 충돌을 방지하기 위해 독립된 가상환경을 만듭니다.

```Bash
python3 -m venv venv
```

#### 3. 가상환경 활성화 (Activate)
생성한 가상환경을 현재 터미널 세션에 적용합니다. (활성화되면 터미널 앞에 (venv) 표시가 나타납니다.)

```Bash
source venv/bin/activate
```
#### 4. 필수 패키지 설치 (Install Dependencies)
requirements.txt에 기록된 필요한 라이브러리들을 한꺼번에 설치합니다.

```Bash
pip install -r requirements.txt
```

가상환경 종료: 작업을 마친 후 가상환경을 나가려면 deactivate를 입력하세요.

패키지 업데이트: requirements.txt 내용이 변경되었다면 다시 pip install -r requirements.txt를 실행하여 업데이트하세요.