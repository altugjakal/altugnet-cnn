import torch
from colorama import Fore
from models import ImageClassifier, Denoiser
from data_loader import get_data_loaders
from training import train_classifier, train_denoiser
from testing import test_clean_classification, test_adversarial_classification


def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_loader, test_loader = get_data_loaders(batch_size=32)
    
    # Initialize models
    classifier = ImageClassifier().to(device)
    denoiser = Denoiser().to(device)
    
    # Banner
    print(Fore.LIGHTGREEN_EX + """
        _ _                          _
       | | |                        | |
   __ _| | |_ _   _  __ _ _ __   ___| |_
  / _` | | __| | | |/ _` | '_ \ / _ \ __|
 | (_| | | |_| |_| | (_| | | | |  __/ |_
  \__,_|_|\__|\__,_|\__, |_| |_|\___|\__| v0.0.1
                     __/ |
                    |___/
""")
    
    # Train models
    train_classifier(classifier, train_loader, device)
    train_denoiser(denoiser, classifier, train_loader, device)
    
    # Command loop
    is_denoiser_on = False
    print(Fore.LIGHTGREEN_EX + "Type exit to exit; type test, denoiser, or aa to continue: " + Fore.RESET)
    
    while True:
        command = input().strip().lower()
        
        if command == "test":
            print(Fore.LIGHTGREEN_EX + "Testing model")
            test_clean_classification(classifier, test_loader, device)
            print(Fore.LIGHTGREEN_EX + "\nModel tested" + Fore.RESET)
            
        elif command == "aa":
            print(Fore.LIGHTGREEN_EX + "Testing adversarial attack")
            test_adversarial_classification(classifier, denoiser, test_loader, device, is_denoiser_on)
            print(Fore.LIGHTGREEN_EX + "\nAdversarial attack complete" + Fore.RESET)
            
        elif command == "denoiser":
            is_denoiser_on = not is_denoiser_on
            status = "on" if is_denoiser_on else "off"
            print(Fore.LIGHTGREEN_EX + f"Denoiser turned {status}" + Fore.RESET)
            
        elif command == "exit":
            print(Fore.LIGHTGREEN_EX + "Exiting")
            break
            
        else:
            print(Fore.LIGHTGREEN_EX + "Invalid command" + Fore.RESET)


if __name__ == "__main__":
    main()