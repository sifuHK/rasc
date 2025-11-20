# -*- coding: utf-8 -*-

from rascpy.ScoreCard import CardFlow
# Windows requires a main function to be written (but not in Jupyter), while Linux and macOS can omit the main function.
if __name__ == '__main__':
    # Pass in the command file
    scf = CardFlow('./inst_01.txt')
    scf.start(start_step=1,end_step=10)
    
    # start_step and end_step can also be omitted, abbreviated as:
    # scf.start(1,10) 
    
    # There are 11 steps: 1. load data, 2. Equal-frequency binning, 3. Variable pre-filtering, 4. Monotonicity suggestion, 5. Optimal binning, 6. WOE transformation, 7. Variable filtering, 8. Modeling, 9. Generate scorecard, 10. Output model report, 11. Reject inference scorecard development.
    # scf.start(start_step=1,end_step=11)#Generate scorecard + reject inference scorecard, totaling two scorecards.

    # You can stop at any step, as follows:
    # scf.start(start_step=1,end_step=10)#Do not develop a reject inference scorecard.
    # scf.start(start_step=1,end_step=9)#Do not output model report.
    
    # For completed results, if there are no modifications to the related instructions, they do not need to be run again. As shown below, steps 1-4 that have already been completed will be automatically loaded (unaffected by computer restart).
    # scf.start(start_step=5,end_step=8)