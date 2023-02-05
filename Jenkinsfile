pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                echo 'Building SD2 Server'
                sh '''
                export PATH=/home/ubuntu/main_drive/anaconda3/bin:$PATH # modify this path 
                eval "$(conda shell.bash hook)"
                conda activate ldm
                pip install -r requirements.txt
                echo 'Build complete'
                '''
                
            }
        }
        stage('Deliver') {
            steps {
                echo 'Copying Files to deploy zone'
                sh '''
                echo "doing delivery stuff.."
                ls
                '''
            }
        }
    }
}