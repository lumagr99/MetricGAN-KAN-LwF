# MetricGAN-KAN
Combine MetricGAN+ with KAN.
# Requirements
```bash
pip install speechbrain
pip install https://github.com/schmiph2/pysepm/archive/master.zip
```

If `kaiser` is not found leading to an `ImportError`, you may need to go to line 2 of /path/to/your/python_env/site-packages/pysepm/utils.py, and change
```python
from scipy.signal import firls,kaiser,upfirdn
```
to
```python
from scipy.signal import firls,upfirdn
from scipy.signal.windows import kaiser
```

# References
## efficient_kan
```
https://github.com/Blealtan/efficient-kan
```
## kan_convs and kans
```
https://github.com/IvanDrokin/torch-conv-kan/tree/main
```

# Paper
```
@INPROCEEDINGS{mgk2025,
  author={Yemin Mai and Stefan Goetze},
  title={{MetricGAN+KAN: Kolmogorov-Arnold Networks in Metric-Driven Speech Enhancement Systems}}, 
  booktitle={Proc.\ Int.\ Conf.\ on Acoustics, Speech, and Signal Processing (ICASSP'25)}, 
  address={Hyderabad, India},
  month= jun,
  year={2025},
}
```