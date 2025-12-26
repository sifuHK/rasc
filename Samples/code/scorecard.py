# -*- coding: utf-8 -*-
from rascpy.ScoreCard import CardFlow

if __name__ == '__main__':# Windows必须要写main函数（但是在jupyter中不需要），Linux和MacOS可以不写main函数
    # 传入指令文件
    scf = CardFlow('./inst_01.txt')
    scf.start(start_step=1,end_step=10)
    
    # 也可以省略start_step和end_step，简写为：
    # scf.start(1,10) 
    
    # 共有11个步骤:1 读取数据，2 等频分箱，3 变量预过滤，4 单调性建议，5 最优分箱，6 WOE转换，7 变量过滤，8 建模，9 生成评分卡，10 输出模型报告，11 拒绝推断评分卡开发
    # scf.start(start_step=1,end_step=11)#生成评分卡+拒绝推断评分卡，共两个评分卡  

    # 你可以在任意步停止，如下：
    # scf.start(start_step=1,end_step=10)#不会开发拒绝推断的评分卡
    # scf.start(start_step=1,end_step=9)#不会输出模型报告
    
    # 已经运行完的结果，如果没有修改与其有关的指令，则不需要再次运行。如下，会自动加载已经运行完的1-4步（不会受到重启计算机的影响）
    # scf.start(start_step=5,end_step=8)