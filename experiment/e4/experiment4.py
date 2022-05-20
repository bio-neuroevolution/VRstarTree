import logging
import csv
import matplotlib.pyplot as plt

import simulation
from blocks import BlockChain
from dataloader import PocemonLoader
from log import LogHandler
from merklePatriciaTree.patricia_tree import MerklePatriciaTree
from run import _initdata
from utils import Configuration, Collections

def experiment4():
    """
    比较MPT树和Verkle树，Merkle KD-Tree和两个Verkle AR*-tree树的证明长度
    """
    context = Configuration(max_children_num=8, max_entries_num=8, account_length=8, account_count=200,
                            select_nodes_func='', merge_nodes_func='', split_node_func='')
    query_param = dict(count=200, sizes=[2, 2, 2], posrandom=100, lengthcenter=0.05, lengthscale=0.1)
    blocksizes = [30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250]
    account_lengths,tran_lengths,traj_lengths,block_dag_lengths,mpt_lengths = [],[],[],[],[]

    proof_length_unit ,proof_sample_count= 64,10
    # 读取数据
    transactions = _initdata(context, {})

    for i,blocksize in enumerate(blocksizes):
        # 创建区块链
        logging.info('创建BlockChain...')
        chain = BlockChain(context, blocksize, transactions=transactions)
        # 计算叶节点的证明长度
        account_length,tran_length,traj_length = chain.proof_length(count=proof_sample_count,unit=proof_length_unit)
        print(str(account_length)+","+str(tran_length)+","+str(traj_length))

        logging.info('创建BlockDAG...')
        settings = dict(repeat_times=1, tr=60, D=3, bs=blocksize, alpha=10)
        block_dag = simulation.GeneratorDAGchain.generate(**settings)
        block_dag_length = block_dag.merkle_kd_trees[1].proof_length(proof_length_unit)
        print(block_dag_length)


        logging.info('创建MPT...')
        dataLoader = PocemonLoader()
        account_names, _ = dataLoader.create_account_name(blocksize, context.account_length)
        mpt = MerklePatriciaTree(xh=i+1,from_scratch=True)

        for name in account_names:
            mpt.insert(name)
        mpt_length = mpt.proof_length(count=proof_sample_count,unit=proof_length_unit)
        print(mpt_length)
        #import shutil
        #shutil.rmtree('./db')

        account_lengths.append(account_length)
        tran_lengths.append(tran_length)
        traj_lengths.append(tran_length)
        block_dag_lengths.append(block_dag_length)
        mpt_lengths.append(mpt_length)

    logging.info('实验四结果：')
    logging.info(blocksizes)
    logging.info(account_lengths)
    logging.info(tran_lengths)
    logging.info(traj_lengths)
    logging.info(block_dag_lengths)
    logging.info(mpt_lengths)

    log_path = 'experiment4.csv'
    file = open(log_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(blocksizes)
    csv_writer.writerow(account_lengths)
    csv_writer.writerow(tran_lengths)
    csv_writer.writerow(traj_lengths)
    csv_writer.writerow(block_dag_lengths)
    csv_writer.writerow(mpt_lengths)
    file.close()

    plt.figure(7)
    plt.plot(blocksizes, account_lengths,label="Account")
    plt.plot(blocksizes, tran_lengths,label="Transaction")
    plt.plot(blocksizes, traj_lengths,label="Trajectory")
    plt.plot(blocksizes, block_dag_lengths,label='BlockDAG')
    plt.plot(blocksizes, mpt_lengths,label='MPT')
    plt.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75', dashes=(15, 10))
    plt.xlabel('Block size')
    plt.ylabel('Proof length')
    plt.legend(loc='best')
    plt.savefig('experiment4_proof.png')

if __name__ == '__main__':
    experiment4()