import os
import glob

from federatedscope.core.communication import gRPCCommManager
from federatedscope.core.message import Message
from omegaconf import OmegaConf

from .aggregator import callbacks

class Server(object):
    def __init__(self, client_num, fed_type):
        self.client_num = client_num
        self.manager = gRPCCommManager(
            host = "127.0.0.1", 
            port = "55555", 
            client_num = self.client_num)
        self.clients = []

        self.fed_func = callbacks[fed_type].fed_func

    def join_clients(self):
        while len(self.clients) < self.client_num:
            msg = self.manager.receive()
            if msg.msg_type == "join":
                self.manager.add_neighbors(
                    neighbor_id = msg.sender, 
                    address = {
                        "host": msg.content.split(":")[0], 
                        "port": msg.content.split(":")[1]})
                self.clients.append(msg.sender)
        self.manager.send(Message(
            msg_type = "start", 
            sender = 0, 
            receiver = self.clients, 
            content = "start"))

    def communicate_weight(self):
        end = False
        while True:
            contents = []
            while len(contents) < self.client_num:
                msg = self.manager.receive()
                if msg.msg_type == "client_weight":
                    contents.append(msg.content)
                elif msg.msg_type == "end":
                    end = True
                    break
            if end: break
            content = self.fed_func(contents)
            self.manager.send(Message(
                msg_type = "host_weight", 
                sender = 0, 
                receiver = self.clients, 
                content = content))

    def run(self):
        # self.start_clients()
        self.join_clients()
        self.communicate_weight()