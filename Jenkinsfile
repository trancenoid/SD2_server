pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                echo 'Building SD2 Server'
                conda activate ldm
                conda install --file requirements.txt
                echo 'Build complete'
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