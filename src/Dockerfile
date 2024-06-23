FROM nvcr.io/nvidia/pytorch:20.12-py3
    
    LABEL maintainer="izbassaramina@gmail.com"
    ENV TZ=Asia/Almaty \
        USER=amina_izbassar \
        UID=1012
    
    RUN groupadd -g ${UID} ${USER} && useradd -l -r -m -s /bin/bash -u ${UID} ${USER} -g ${USER}
    
    #Install your required software
    # RUN apt update -y && \
    #     apt -y install ca-certificates tzdata software-properties-common cmake
            
    USER ${USER}
    
    WORKDIR /home/amina_izbassar/
    COPY . .
            
    RUN python3 -m venv yolov8n && source yolov8n/bin/activate
            
    RUN python3 -m pip install --upgrade pip 
    RUN pip3 install -U -r requirements.txt