# Accessing the NPU from inside a Docker container

First, install the [Container Device Interface (CDI)](https://docs.docker.com/build/building/cdi/) for your Dragonwing development board via:

```bash
# RB3 Gen 2 Vision Kit / Rubik Pi 3
sudo python3 cdi/install_cdi.py --file cdi/cdi-hw-acc-6490.json

# IQ-9075 EVK
sudo python3 cdi/install_cdi.py --file cdi/cdi-hw-acc-9100.json
```

Now you can just pass `--device qualcomm.com/device=cdi-hw-acc` when you run a container to get NPU access.

Here's a demo of using a LiteRT/TFLite model on the NPU:

```bash
cd litert
docker build -t litert-demo .
docker run -it --device qualcomm.com/device=cdi-hw-acc \
    -v $PWD:/app \
    litert-demo \
    bash -c "/venvs/litert-demo/bin/python3 run_inference.py --use-npu"

# Top-5 predictions:
# Class boa constrictor: score=0.39735516905784607
# Class rock python: score=0.22408385574817657
# Class night snake: score=0.019640149548649788
# Class eggnog: score=0.002774509834125638
# Class cup: score=0.0019864204805344343
#
# Inference took (on average): 126.8ms. per image
```

> Omit `--use-npu` to run the same model on the CPU.
