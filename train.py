import model
import reader
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import copy
import yaml
import math
import time

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    config = yaml.load(open("config.yaml"))
    config = config["train"]
    path = config["data"]["path"]
    model_name = config["save"]["model_name"]


    save_path = os.path.join(config["save"]["save_path"], "checkpoint")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Read data")
    dataset = reader.txtload(path, "train", config["params"]["batch_size"], shuffle=True,
                             num_workers=0)

    print("Model building")
    net = model.model()

    net.train()
    net = nn.DataParallel(net)
    #state_dict = torch.load(os.path.join(save_path, f"Iter_10_best_flip.pt"))
    #net.load_state_dict(state_dict)
    net.to(device)

    print("optimizer building")
    loss_op = nn.SmoothL1Loss().cuda()
    base_lr = config["params"]["lr"]
    cur_step = 0
    decay_steps = config["params"]["decay_step"]
    optimizer = torch.optim.Adam(net.parameters(), base_lr,
                                weight_decay=0.0005)
    print("Traning")
    length = len(dataset)
    cur_decay_index = 0
    with open(os.path.join(save_path, "train_log"), 'w') as outfile:
        for epoch in range(1, config["params"]["epoch"] + 1):
            if cur_decay_index < len(decay_steps) and epoch == decay_steps[cur_decay_index]:
                base_lr = base_lr * config["params"]["decay"]
                cur_decay_index = cur_decay_index + 1
                for param_group in optimizer.param_groups:
                    param_group["lr"] = base_lr
            #if (epoch <= 10):
            #    continue

            time_begin = time.time()
            for i, (data) in enumerate(dataset):

                data["faceImg"] = data["faceImg"].to(device)
                data["leftEyeImg"] = data["leftEyeImg"].to(device)
                data['rightEyeImg'] = data['rightEyeImg'].to(device)
                data['rects'] = data['rects'].to(device)
                label = data["label"].to(device)
                # print(data["face"].shape)
                # print(data["left"].shape)
                # print(data['head_pose'].shape)
                gaze = net(data["leftEyeImg"], data["rightEyeImg"], data['faceImg'], data['rects'])
                loss = loss_op(gaze, label)*4

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                time_remain = (length-i-1) * ((time.time()-time_begin)/(i+1)) /  3600   #time estimation for current epoch
                epoch_time = (length-1) * ((time.time()-time_begin)/(i+1)) / 3600       #time estimation for 1 epoch
                #person_time = epoch_time * (config["params"]["epoch"])                  #time estimation for 1 subject
                time_remain_total = time_remain + \
                                    epoch_time * (config["params"]["epoch"]-epoch)
                                    #person_time * (len(subjects) - subject_i - 1) 
                log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss:.5f} lr:{base_lr} time:{time_remain:.2f}h total:{time_remain_total:.2f}h"
                outfile.write(log + "\n")
                # if i % 20 == 0:
                print(log)
                sys.stdout.flush()
                outfile.flush()

            if epoch % config["save"]["step"] == 0:
                torch.save(net.state_dict(), os.path.join(save_path, f"Iter_{epoch}_{model_name}.pt"))

