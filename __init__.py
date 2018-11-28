# do this so "import dlct" works for modules within the package (seems hacky)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Want to be able to call delectable.train_model.train_model() as delectable.train_model()
from delectable.train_model import train_model
