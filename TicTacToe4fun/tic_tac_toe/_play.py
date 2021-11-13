# -*- coding: utf-8 -*-

# Author: Tank Overlord <TankOverLord88@gmail.com>
#
# License: MIT

import random
import time

"""
Key takeaway: you need to visit all the terminal leaf nodes in order to get the global true scores of the parent nodes.
"""

class play():

    def __init__(self):
        self.use_alpha_beta_pruning = True
        self.maximizer_score_history = {}
        self.minimizer_score_history = {}
        self.X_all_scores_history = {}
        self.O_all_scores_history = {}

    def find_empty_squares(self, B: dict = None):
        empty_squares = []
        for key in B:
            if B[key] == ' ':
                empty_squares.append(key)
        return empty_squares

    def printboard(self, B: dict = None):
        print(f"|{B[1]}{B[2]}{B[3]}|\n"
            f"|{B[4]}{B[5]}{B[6]}|\n"
            f"|{B[7]}{B[8]}{B[9]}|\n")

    def checkwin(self, B: dict = None):
        """
        1 = X won
        0 = Draw
        -1 = X won
        None = not terminal state
        """
        for r in [1,4,7]:
            this_row = (B[r], B[r+1], B[r+2])
            if this_row == ('X', 'X', 'X'):
                return 1
            elif this_row == ('O', 'O', 'O'):
                return -1
        for c in [1,2,3]:
            this_col = (B[c], B[c+3], B[c+6])
            if this_col == ('X', 'X', 'X'):
                return 1
            elif this_col == ('O', 'O', 'O'):
                return -1
        this_diagnoal_1 = (B[1], B[5], B[9])
        this_diagnoal_2 = (B[3], B[5], B[7])
        if this_diagnoal_1 == ('X', 'X', 'X') or this_diagnoal_2 == ('X', 'X', 'X'):
            return 1
        elif this_diagnoal_1 == ('O', 'O', 'O') or this_diagnoal_2 == ('O', 'O', 'O'):
            return -1
        for i in range(1, 10):
            if B[i] == ' ':
                return None
        return 0

    def minimax_score(self, B: dict = None, alpha = float('-inf'), beta = float('+inf'), isMaximizing: bool = True):
        score = self.checkwin(B)
        if score is not None:
            return score
        str_B = str(B)
        if isMaximizing: # X ("alpha" player) plays
            if str_B not in self.maximizer_score_history:
                best_score = float('-inf')
                for square in self.find_empty_squares(B):
                    B[square] = 'X'
                    score = self.minimax_score(B = B, alpha = alpha, beta = beta, isMaximizing = False)
                    B[square] = ' '
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                    if self.use_alpha_beta_pruning and beta <= alpha:
                        if best_score == 1: # this is to save the alpha beta pruning when used in conjunction with hashmap. if the "best_score" cannot be better, it must be the best_score
                            break # parent beta cutoff
                self.maximizer_score_history[str_B] = {'alpha': alpha, 'beta': beta, 'best_score': best_score}
            return self.maximizer_score_history[str_B]['best_score']
        else: # O ("beta" player) plays
            if str_B not in self.minimizer_score_history:
                best_score = float('+inf')
                for square in self.find_empty_squares(B):
                    B[square] = 'O'
                    score = self.minimax_score(B = B, alpha = alpha, beta = beta, isMaximizing = True)
                    B[square] = ' '
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    if self.use_alpha_beta_pruning and beta <= alpha:
                        if best_score == -1: # this is to save the alpha beta pruning when used in conjunction with hashmap. if the "best_score" cannot be better, it must be the best_score
                            break # parent alpha cutoff
                self.minimizer_score_history[str_B] = {'alpha': alpha, 'beta': beta, 'best_score': best_score}
            return self.minimizer_score_history[str_B]['best_score']

    def minimax_vs_minimax(self, verbosity = 0):
        B = {1: ' ', 2: ' ', 3: ' ',
            4: ' ', 5: ' ', 6: ' ',
            7: ' ', 8: ' ', 9: ' '}
        if verbosity == 1:
            self.printboard(B)
        if random.randint(1,2) == 1:
            turn = 'X'
        else:
            turn = 'O'
        while self.checkwin(B) is None:
            str_B = str(B)
            if turn == 'X':
                if str_B not in self.X_all_scores_history:
                    X_all_scores = {}
                    X_best_score = float('-inf')
                    for square in self.find_empty_squares(B):
                        B[square] = 'X'
                        score = self.minimax_score(B, isMaximizing = False) # see what score 'X' can still have in this situation, assuming 'O' plays optimally next
                        B[square] = ' '
                        if score in X_all_scores:
                            X_all_scores[score].append(square)
                        else:
                            X_all_scores[score] = [square,]
                        X_best_score = max(score, X_best_score)
                    self.X_all_scores_history[str_B] = {'X_all_scores': X_all_scores, 'X_best_score': X_best_score}
                this_dict = self.X_all_scores_history[str_B]
                X_all_scores = this_dict['X_all_scores']
                X_best_score = this_dict['X_best_score']
                xmove = random.sample(X_all_scores[X_best_score], 1)[0]
                B[xmove] = 'X'
                turn = 'O'
            else:
                if str_B not in self.O_all_scores_history:
                    O_all_scores = {}
                    O_best_score = float('+inf')
                    for square in self.find_empty_squares(B):
                        B[square] = 'O'
                        score = self.minimax_score(B, isMaximizing = True) # see what score 'O' can still have in this situation, assuming 'X' plays optimally next
                        B[square] = ' '
                        if score in O_all_scores:
                            O_all_scores[score].append(square)
                        else:
                            O_all_scores[score] = [square,]
                        O_best_score = min(score, O_best_score)
                    self.O_all_scores_history[str_B] = {'O_all_scores': O_all_scores, 'O_best_score': O_best_score}
                this_dict = self.O_all_scores_history[str_B]
                O_all_scores = this_dict['O_all_scores']
                O_best_score = this_dict['O_best_score']                
                omove = random.sample(O_all_scores[O_best_score], 1)[0]
                B[omove] = 'O'
                turn = 'X'
            if verbosity == 1:
                self.printboard(B)
        return self.checkwin(B)

    def trials(self, verbosity = 0, n_trials = 10000):
        x_won, o_won, draw = 0, 0, 0
        start = time.time()
        for i in range(1, n_trials+1):
            if verbosity == 1:
                print(f"--------\nGame #{i}\n")
            res = self.minimax_vs_minimax(verbosity = verbosity)
            if res == 1:
                x_won += 1
            elif res == -1:
                o_won += 1
            else:
                draw += 1
        end = time.time()
        print(f"X won #: {x_won}, O won #: {o_won}, Draw #: {draw:,d}, Elapsed time: {end - start:.3f} sec")
