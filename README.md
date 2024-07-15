# Be-mine(or)

This is the repository of the art installation called Be-mine(or) by (Marvin) Jerome Stephan.

## Table of Contents
- [Be-mine(or)](#be-mineor)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)

## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have installed Python 3.9.6. You can download it from the [official Python website](https://www.python.org/downloads/release/python-396/).
- You have installed `git` to clone the repository. You can download it from the [official Git website](https://git-scm.com/downloads).

## Installation

To set up the project, follow these steps:

1. **Clone the Repository:**
    ```sh
    git clone https://github.com/HerzogVonWiesel/Be-mine-or.git
    cd Be-mine-or
    ```

2. **Create a Virtual Environment:**
    ```sh
    python3.9 -m venv bemineor
    ```

3. **Activate the Virtual Environment:**
    - On Windows:
        ```sh
        bemineor\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source bemineor/bin/activate
        ```

4. **Install the Required Packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

The project has been optimized to run on Intel hardware using [OpenVINO™](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) (Open Visual Inference and Neural Network Optimization) to enhance performance. You can run this version using

```sh
python be_mine_or_openvino.py
```
or run the standard version using
```sh
python be_mine_or.py
```
