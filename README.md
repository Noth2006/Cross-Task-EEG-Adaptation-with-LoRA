# Cross Task EEG Adaptation with LoRA

This repo consists of a EEGnet 8,2 model completely trained on Motor Execution (ME) adapted to Motor Imagery (MI) using LoRA. 

- First the Model is trained on ME data. A different model per subject is trained. Each subject is trained across 5 seeds. 

- The ME-trained model is frozen, including its BatchNorm statistics. LoRA adapters are added to the last pointwise conv  and the final linear layer, and trained only on offline MI data, amounting to 230 parameters. 

- Test accuracy (majority vote), averaged over 5 seeds and 21 subjects on the held out Motor Imagery set is 44% compared to 46% acheived when the full model was trained on the Motor Imagery data.

- Data is from :  Ding et al., EEG-based brain-computer interface enables real-time robotic hand control at individual finger level, Nature Communications 2025
