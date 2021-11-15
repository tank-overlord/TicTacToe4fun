# -*- coding: utf-8 -*-

# Author: Tank Overlord <TankOverLord88@gmail.com>
#
# License: MIT

import random
import time

"""
Key takeaway: you need to visit all the terminal leaf nodes in order to get the global best scores.
"""

class game():

    def __init__(self):
        self.maximizer_score_history, self.minimizer_score_history = {}, {}
        self.X_all_scores_history, self.O_all_scores_history = {}, {}
        self.use_alpha_beta_pruning = True
        self.use_hashmap = True
        self.test_early_draw = False # when horizontal, vertical, diagnoal has OX, thus impossible to win (not much help with memory though)

    def find_empty_squares(self, B: dict = None):
        return [key for key in B if B[key] == ' ']

    def printboard(self, B: dict = None):
        """
        board_dims = (r, c)
        """
        for r in range(1, self.board_dims[0]+1):
            print("|", end = '')
            for c in range(1, self.board_dims[1]+1):
                print(f"{B[(r,c)]}", end = '')
            print("|\n", end = "")
        print("")

    def checkwin(self, B: dict = None):
        """
        1 = X won
        0 = Draw
        -1 = O won
        None = not terminal state

        board_dims = (r, c)
        """
        """
        # check horizontal
        for r in range(1, self.board_dims[0]+1):
            all_X, all_O = True, True
            for c in range(1, self.board_dims[1]+1):
                if B[(r, c)] == 'X':
                    all_O = False
                elif B[(r, c)] == 'O':
                    all_X = False
                else:
                    all_O, all_X = False, False
                    break
            if all_X:
                return 1
            elif all_O:
                return -1
        # check vertical
        for c in range(1, self.board_dims[1]+1):
            all_X, all_O = True, True
            for r in range(1, self.board_dims[0]+1):
                if B[(r, c)] == 'X':
                    all_O = False
                elif B[(r, c)] == 'O':
                    all_X = False
                else:
                    all_O, all_X = False, False
                    break
            if all_X:
                return 1
            elif all_O:
                return -1
        # check diagnoal "\"
        all_X, all_O = True, True
        for r in range(1, self.board_dims[0]+1):
            if B[(r, r)] == 'X':
                all_O = False
            elif B[(r, r)] == 'O':
                all_X = False
            else: # B[(r, c)] = ' '
                all_O, all_X = False, False
                break
        if all_X:
            return 1
        elif all_O:
            return -1
        # check diagnoal "/"
        all_X, all_O = True, True
        for r in range(self.board_dims[0], 0, -1):   # r = 3, 2, 1, c = 1, 2, 3
            if B[(r, self.board_dims[0]-r+1)] == 'X':
                all_O = False
            elif B[(r, self.board_dims[0]-r+1)] == 'O':
                all_X = False
            else: # B[(r, self.board_dims[0]-r+1)] == ' '
                all_O, all_X = False, False
                break
        if all_X:
            return 1
        elif all_O:
            return -1
        #
        """
        if self.board_dims[0] == 3:
            for triplet in (((1,1),(1,2),(1,3)),((2,1),(2,2),(2,3)),((3,1),(3,2),(3,3)),
                            ((1,1),(2,1),(3,1)),((1,2),(2,2),(3,2)),((1,3),(2,3),(3,3)),
                            ((1,1),(2,2),(3,3)),((1,3),(2,2),(3,1))):
                if B[triplet[0]] == 'X' and B[triplet[1]] == 'X' and B[triplet[2]] == 'X':
                    return 1
                elif B[triplet[0]] == 'O' and B[triplet[1]] == 'O' and B[triplet[2]] == 'O':
                    return -1
        elif self.board_dims[0] == 4:
            for quartet in (((1,1),(1,2),(1,3),(1,4)),((2,1),(2,2),(2,3),(2,4)),((3,1),(3,2),(3,3),(3,4)),((4,1),(4,2),(4,3),(4,4)),
                            ((1,1),(2,1),(3,1),(4,1)),((1,2),(2,2),(3,2),(4,2)),((1,3),(2,3),(3,3),(4,3)),((1,4),(2,4),(3,4),(4,4)),
                            ((1,1),(2,2),(3,3),(4,4)),((1,4),(2,3),(3,2),(4,1))):
                if B[quartet[0]] == 'X' and B[quartet[1]] == 'X' and B[quartet[2]] == 'X' and B[quartet[3]] == 'X':
                    return 1
                elif B[quartet[0]] == 'O' and B[quartet[1]] == 'O' and B[quartet[2]] == 'O' and B[quartet[3]] == 'O':
                    return -1
        else:
            print('Error: dims < (3, 3) or dims > (4, 4) ==> undefined')
            exit(1)
        if self.test_early_draw:
            non_draw_sets_comprehensive_list = [{'X', ' '}, {'O', ' '}, {' ',}]
            if self.board_dims[0] == 3:
                for triplet in (((1,1),(1,2),(1,3)),((2,1),(2,2),(2,3)),((3,1),(3,2),(3,3)),
                                ((1,1),(2,1),(3,1)),((1,2),(2,2),(3,2)),((1,3),(2,3),(3,3)),
                                ((1,1),(2,2),(3,3)),((1,3),(2,2),(3,1))):
                    this_set = set([B[triplet[0]], B[triplet[1]], B[triplet[2]]])
                    if this_set in non_draw_sets_comprehensive_list:
                        return None
                return 0
            elif self.board_dims[0] == 4:
                for quartet in (((1,1),(1,2),(1,3),(1,4)),((2,1),(2,2),(2,3),(2,4)),((3,1),(3,2),(3,3),(3,4)),((4,1),(4,2),(4,3),(4,4)),
                                ((1,1),(2,1),(3,1),(4,1)),((1,2),(2,2),(3,2),(4,2)),((1,3),(2,3),(3,3),(4,3)),((1,4),(2,4),(3,4),(4,4)),
                                ((1,1),(2,2),(3,3),(4,4)),((1,4),(2,3),(3,2),(4,1))):
                    this_set = set([B[quartet[0]], B[quartet[1]], B[quartet[2]], B[quartet[3]]])
                    if this_set in non_draw_sets_comprehensive_list:
                        return None
                return 0
        if len(self.find_empty_squares(B = B)) == 0: # this must be placed here, after checking XXX and OOO, because X or O can have won when no more available moves
            return 0 # Draw
        return None # not terminal state

    def minimax_score(self, B: dict = None, alpha = float('-inf'), beta = float('+inf'), isMaximizing: bool = True):
        """
        board_dims = (r, c)
        """
        score = self.checkwin(B = B)
        if score is not None:
            return score
        str_B = str(B)
        if isMaximizing: # X ("alpha" player) plays
            if self.use_hashmap:
                if str_B not in self.maximizer_score_history:
                    best_score = float('-inf')
                    available_squares = self.find_empty_squares(B = B)
                    available_squares = random.sample(available_squares, len(available_squares))
                    for square in available_squares:
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
            else:
                best_score = float('-inf')
                available_squares = self.find_empty_squares(B = B)
                available_squares = random.sample(available_squares, len(available_squares))
                for square in available_squares:
                    B[square] = 'X'
                    score = self.minimax_score(B = B, alpha = alpha, beta = beta, isMaximizing = False)
                    B[square] = ' '
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                    if self.use_alpha_beta_pruning and beta <= alpha:
                        break # parent beta cutoff
                return best_score
        else: # O ("beta" player) plays
            if self.use_hashmap:
                if str_B not in self.minimizer_score_history:
                    best_score = float('+inf')
                    available_squares = self.find_empty_squares(B = B)
                    available_squares = random.sample(available_squares, len(available_squares))
                    for square in available_squares:
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
            else:
                best_score = float('+inf')
                available_squares = self.find_empty_squares(B = B)
                available_squares = random.sample(available_squares, len(available_squares))
                for square in available_squares:
                    B[square] = 'O'
                    score = self.minimax_score(B = B, alpha = alpha, beta = beta, isMaximizing = True)
                    B[square] = ' '
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    if self.use_alpha_beta_pruning and beta <= alpha:
                        break # parent alpha cutoff
                return best_score

    def minimax_vs_minimax(self, verbosity = 0):
        """
        board_dims = (r, c)
        """
        B = {}
        for r in range(1, self.board_dims[0]+1):
            for c in range(1, self.board_dims[1]+1):
                B[(r,c)] = ' '
        #if verbosity == 1:
        #    self.printboard(B)
        """
        if random.randint(1,2) == 1:
            turn = 'X'
        else:
            turn = 'O'
        """
        turn = 'X'
        while self.checkwin(B) is None:
            str_B = str(B)
            if turn == 'X':
                if self.use_hashmap:
                    if str_B not in self.X_all_scores_history:
                        X_all_scores = {}
                        X_best_score = float('-inf')
                        for square in self.find_empty_squares(B):
                            B[square] = 'X'
                            score = self.minimax_score(B = B, isMaximizing = False) # see what score 'X' can still have in this situation, assuming 'O' plays optimally next
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
                else:
                    X_best_score = float('-inf')
                    available_squares = self.find_empty_squares(B)
                    available_squares = random.sample(available_squares, len(available_squares))
                    for square in available_squares:
                        B[square] = 'X'
                        score = self.minimax_score(B = B, isMaximizing = False) # see what score 'X' can still have in this situation, assuming 'O' plays optimally next
                        B[square] = ' '
                        if score > X_best_score:
                            X_best_score = score
                            xmove = square
                B[xmove] = 'X'
                turn = 'O'
            else:
                if self.use_hashmap:
                    if str_B not in self.O_all_scores_history:
                        O_all_scores = {}
                        O_best_score = float('+inf')
                        for square in self.find_empty_squares(B):
                            B[square] = 'O'
                            score = self.minimax_score(B = B, isMaximizing = True) # see what score 'O' can still have in this situation, assuming 'X' plays optimally next
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
                else:
                    O_best_score = float('+inf')
                    available_squares = self.find_empty_squares(B)
                    available_squares = random.sample(available_squares, len(available_squares))
                    for square in available_squares:
                        B[square] = 'O'
                        score = self.minimax_score(B = B, isMaximizing = True) # see what score 'O' can still have in this situation, assuming 'X' plays optimally next
                        B[square] = ' '
                        if score < O_best_score:
                            O_best_score = score       
                            omove = square
                B[omove] = 'O'
                turn = 'X'
            if verbosity == 1:
                self.printboard(B = B)
        return self.checkwin(B = B)

    def trials(self, verbosity = 0, n_trials = 10000, board_dims = (3, 3)):
        """
        board_dims = (r, c)
        """
        if board_dims[0] != board_dims[1]:
            print('Error: it must be a square board')
            exit(1)
        else:
            self.board_dims = board_dims
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
        print(f"board_dims = {self.board_dims}, X won #: {x_won}, O won #: {o_won}, Draw #: {draw:,d}, Elapsed time: {end - start:.3f} sec")
