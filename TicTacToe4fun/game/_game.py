# -*- coding: utf-8 -*-

# Author: Tank Overlord <TankOverLord88@gmail.com>
#
# License: MIT

import sys
import random
import time
import tracemalloc
import pickle
import zlib
import importlib.resources as pkg_resources
from .. import hashmap

"""
Key takeaway: you need to visit all the terminal leaf nodes in order to get the global best scores.
"""

class tictactoe(object):
    """
    in development: a hopefully much faster version of tic tac toe design

    question: can we have y = row_sums, col_sum, diag_sum, antidiag_sum, and x = (r, c) and we aim for solving an equation without minimax?
    """
    def __init__(self, preload_hashmap = True):
        self.map = {1: 'X', 0: ' ', -1: 'O'}
        if preload_hashmap:
            self.maximizer_best_moves_hashmap = pickle.loads(zlib.decompress(pkg_resources.read_binary(hashmap, 'maximizer_best_moves.gz')))
            self.minimizer_best_moves_hashmap = pickle.loads(zlib.decompress(pkg_resources.read_binary(hashmap, 'minimizer_best_moves.gz')))
        else:
            self.maximizer_best_moves_hashmap, self.minimizer_best_moves_hashmap = {}, {}

    def reset(self):
        self.max_moves = self.n * self.n
        self.row_sums = [0] * self.n
        self.col_sums = [0] * self.n
        self.diag_sum = self.antidiag_sum = 0
        self.board = {}
        for r in range(self.n):
            for c in range(self.n):
                self.board[(r, c)] = 0
        self.n_moves = 0

    def build_hashmap(self):
        self.trials(n = 3, verbosity = 0, n_trials = 10000, use_alpha_beta_pruning = True, use_hashmap = True)
        self.trials(n = 4, verbosity = 0, n_trials = 1,     use_alpha_beta_pruning = True, use_hashmap = True)
        with open('./TicTacToe4fun/hashmap/maximizer_best_moves.gz', 'wb') as fp:
            fp.write(zlib.compress(pickle.dumps(self.maximizer_best_moves_hashmap, pickle.HIGHEST_PROTOCOL),9))
            fp.close()
        with open('./TicTacToe4fun/hashmap/minimizer_best_moves.gz', 'wb') as fp:
            fp.write(zlib.compress(pickle.dumps(self.minimizer_best_moves_hashmap, pickle.HIGHEST_PROTOCOL),9))
            fp.close()

    def board_hash(self):
        return ''.join([self.map[x] for x in self.board.values()])
        
    def find_available_moves(self):
        return [key for key in self.board if self.board[key] == 0]

    def printboard(self):
        for r in range(self.n):
            print("|", end = '')
            for c in range(self.n):
                print(f"{self.map[self.board[(r,c)]]}", end = '')
            print("|\n", end = "")
        print("")

    def move(self, square: tuple = None, player: int = None):
        # player: 1 for 'X', -1 for 'O'
        # move
        row, col = square[0], square[1]
        self.board[(row, col)] += player
        self.n_moves += 1
        self.row_sums[row] += player
        self.col_sums[col] += player
        if row == col:
            self.diag_sum += player
        if (row+col) == (self.n-1):
            self.antidiag_sum += player

    def undo_move(self, square: tuple = None, player: int = None):
        # player: 1 for 'X', -1 for 'O'
        # undo move
        row, col = square[0], square[1]
        self.board[(row, col)] -= player
        self.n_moves -= 1
        self.row_sums[row] -= player
        self.col_sums[col] -= player
        if row == col:
            self.diag_sum -= player
        if (row+col) == (self.n-1):
            self.antidiag_sum -= player

    def checkwin(self):
        if self.n == 3:
            sums_array = [self.row_sums[0], self.row_sums[1], self.row_sums[2],                                     self.col_sums[0], self.col_sums[1], self.col_sums[2],                                     self.diag_sum, self.antidiag_sum]
        elif self.n == 4:
            sums_array = [self.row_sums[0], self.row_sums[1], self.row_sums[2], self.row_sums[3],                   self.col_sums[0], self.col_sums[1], self.col_sums[2], self.col_sums[3],                   self.diag_sum, self.antidiag_sum]
        elif self.n == 5:
            sums_array = [self.row_sums[0], self.row_sums[1], self.row_sums[2], self.row_sums[3], self.row_sums[4], self.col_sums[0], self.col_sums[1], self.col_sums[2], self.col_sums[3], self.col_sums[4], self.diag_sum, self.antidiag_sum]
        else:
            print(f'Error: undefined board size: {self.n}x{self.n}. It must be either 3x3 or 4x4.')
            sys.exit(1)
        for this_sum in sums_array:
            if this_sum == self.n:
                return 1
            elif this_sum == -self.n:
                return -1
        if self.n_moves == self.max_moves:
            return 0 # Draw
        return None # not terminal yet

    def minimax_score(self, alpha = float('-inf'), beta = float('+inf'), isMaximizing: bool = None):
        score = self.checkwin()
        if score is not None:
            return score
        board_hash_value = self.board_hash()
        if isMaximizing: # X ("alpha" player) plays
            if self.use_hashmap and board_hash_value in self.maximizer_best_moves_hashmap:
                square = random.sample(self.maximizer_best_moves_hashmap[board_hash_value], 1)[0]
                self.move(square, player = 1)
                best_score = self.minimax_score(alpha = alpha, beta = beta, isMaximizing = False)
                self.undo_move(square, player = 1)
                #best_score = score
                #alpha = max(alpha, best_score)
                #if self.use_alpha_beta_pruning and beta <= alpha:
                #    if self.verbosity >= 2:
                #        print('β cutoff')
                #    break # parent beta cutoff
                return best_score
            else:
                best_score = float('-inf')
                available_moves = self.find_available_moves()
                available_moves = random.sample(available_moves, len(available_moves))
                for square in available_moves:
                    self.move(square, player = 1)
                    score = self.minimax_score(alpha = alpha, beta = beta, isMaximizing = False)
                    self.undo_move(square, player = 1)
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                    if self.use_alpha_beta_pruning and beta <= alpha:
                        if self.verbosity >= 2:
                            print('β cutoff')
                        break # parent beta cutoff
                return best_score
        else: # O ("beta" player) plays
            if self.use_hashmap and board_hash_value in self.minimizer_best_moves_hashmap:
                square = random.sample(self.minimizer_best_moves_hashmap[board_hash_value], 1)[0]
                self.move(square, player = -1)
                best_score = self.minimax_score(alpha = alpha, beta = beta, isMaximizing = True)
                self.undo_move(square, player = -1)
                #best_score = score
                #beta = min(beta, best_score)
                #if self.use_alpha_beta_pruning and beta <= alpha:
                #    if self.verbosity >= 2:
                #        print('α cutoff')
                #    break # parent alpha cutoff
                return best_score
            else:
                best_score = float('+inf')
                available_moves = self.find_available_moves()
                available_moves = random.sample(available_moves, len(available_moves))
                for square in available_moves:
                    self.move(square, player = -1)
                    score = self.minimax_score(alpha = alpha, beta = beta, isMaximizing = True)
                    self.undo_move(square, player = -1)
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    if self.use_alpha_beta_pruning and beta <= alpha:
                        if self.verbosity >= 2:
                            print('α cutoff')
                        break # parent alpha cutoff
                return best_score

    def minimax_vs_minimax(self):
        """
        board_dims = (r, c)
        """
        self.reset()
        turn = 1
        while self.checkwin() is None:
            board_hash_value = self.board_hash()
            if turn == 1:
                if self.use_hashmap:
                    if board_hash_value not in self.maximizer_best_moves_hashmap:
                        score_move_dict = {}
                        X_best_score = float('-inf')
                        available_moves = self.find_available_moves()
                        available_moves = random.sample(available_moves, len(available_moves))
                        for square in available_moves:
                            self.move(square, turn)
                            score = self.minimax_score(isMaximizing = False) # see what score 'X' can still have in this situation, assuming 'O' plays optimally next
                            self.undo_move(square, turn)
                            if score not in score_move_dict:
                                score_move_dict[score] = [square,]
                            else:
                                score_move_dict[score].append(square)
                            if score > X_best_score:
                                X_best_score = score
                        self.maximizer_best_moves_hashmap[board_hash_value] = list(score_move_dict[X_best_score])
                    xmove = random.sample(self.maximizer_best_moves_hashmap[board_hash_value], 1)[0]
                else:
                    X_best_score = float('-inf')
                    available_moves = self.find_available_moves()
                    available_moves = random.sample(available_moves, len(available_moves))
                    for square in available_moves:
                        self.move(square, turn)
                        score = self.minimax_score(isMaximizing = False) # see what score 'X' can still have in this situation, assuming 'O' plays optimally next
                        self.undo_move(square, turn)
                        if X_best_score < score:
                            X_best_score = score
                            xmove = square
                self.move(xmove, turn)
                turn = -1
            else:
                if self.use_hashmap:
                    if board_hash_value not in self.minimizer_best_moves_hashmap:
                        score_move_dict = {}
                        O_best_score = float('+inf')
                        available_moves = self.find_available_moves()
                        available_moves = random.sample(available_moves, len(available_moves))
                        for square in available_moves:
                            self.move(square, turn)
                            score = self.minimax_score(isMaximizing = True) # see what score 'O' can still have in this situation, assuming 'X' plays optimally next
                            self.undo_move(square, turn)
                            if score not in score_move_dict:
                                score_move_dict[score] = [square,]
                            else:
                                score_move_dict[score].append(square)
                            if O_best_score > score:
                                O_best_score = score
                        self.minimizer_best_moves_hashmap[board_hash_value] = list(score_move_dict[O_best_score])
                    omove = random.sample(self.minimizer_best_moves_hashmap[board_hash_value], 1)[0]
                else:
                    O_best_score = float('+inf')
                    available_moves = self.find_available_moves()
                    available_moves = random.sample(available_moves, len(available_moves))
                    for square in available_moves:
                        self.move(square, turn)
                        score = self.minimax_score(isMaximizing = True) # see what score 'O' can still have in this situation, assuming 'X' plays optimally next
                        self.undo_move(square, turn)
                        if score < O_best_score:
                            O_best_score = score       
                            omove = square
                self.move(omove, turn)
                turn = 1
            if self.verbosity >= 1:
                self.printboard()
        return self.checkwin()

    def trials(self, n_trials: int = 1, verbosity: int = 0, use_hashmap: bool = True, use_alpha_beta_pruning: bool = True, n: int = 3):
        self.n = n  # board size
        self.verbosity = verbosity
        self.use_hashmap = use_hashmap
        self.use_alpha_beta_pruning = use_alpha_beta_pruning
        x_won = o_won = draw = 0
        self.start = time.time()
        for i in range(n_trials):
            res = self.minimax_vs_minimax()
            if res == 0:
                draw += 1
            elif res == 1:
                x_won += 1
            else:
                o_won += 1
        self.end = time.time()
        print(f"board size = {self.n}x{self.n}, # of X won: {x_won}, # of O won: {o_won}, # of draw: {draw:,d}, elapsed time: {self.end - self.start:.3f} sec")


#####################################################################################################################


class game():

    def __init__(self):
        self.maximizer_score_history, self.minimizer_score_history = {}, {}
        self.X_all_scores_history, self.O_all_scores_history = {}, {}

    def find_available_moves(self, B: dict = None):
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
        if len(self.find_available_moves(B = B)) == 0: # this must be placed here, after checking XXX and OOO, because X or O can have won when no more available moves
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
                    available_moves = self.find_available_moves(B = B)
                    available_moves = random.sample(available_moves, len(available_moves))
                    for square in available_moves:
                        B[square] = 'X'
                        score = self.minimax_score(B = B, alpha = alpha, beta = beta, isMaximizing = False)
                        B[square] = ' '
                        best_score = max(best_score, score)
                        alpha = max(alpha, best_score)
                        if self.use_alpha_beta_pruning and beta <= alpha:
                            if best_score == 1: # this is to save the alpha beta pruning when used in conjunction with hashmap. if the "best_score" cannot be better, it must be the best_score
                                if self.verbosity >= 2:
                                    print('β cutoff')
                                break # parent beta cutoff
                    self.maximizer_score_history[str_B] = {'alpha': alpha, 'beta': beta, 'best_score': best_score}
                return self.maximizer_score_history[str_B]['best_score']
            else:
                best_score = float('-inf')
                available_moves = self.find_available_moves(B = B)
                available_moves = random.sample(available_moves, len(available_moves))
                for square in available_moves:
                    B[square] = 'X'
                    score = self.minimax_score(B = B, alpha = alpha, beta = beta, isMaximizing = False)
                    B[square] = ' '
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                    if self.use_alpha_beta_pruning and beta <= alpha:
                        if self.verbosity >= 2:
                            print('β cutoff')
                        break # parent beta cutoff
                return best_score
        else: # O ("beta" player) plays
            if self.use_hashmap:
                if str_B not in self.minimizer_score_history:
                    best_score = float('+inf')
                    available_moves = self.find_available_moves(B = B)
                    available_moves = random.sample(available_moves, len(available_moves))
                    for square in available_moves:
                        B[square] = 'O'
                        score = self.minimax_score(B = B, alpha = alpha, beta = beta, isMaximizing = True)
                        B[square] = ' '
                        best_score = min(best_score, score)
                        beta = min(beta, best_score)
                        if self.use_alpha_beta_pruning and beta <= alpha:
                            if best_score == -1: # this is to save the alpha beta pruning when used in conjunction with hashmap. if the "best_score" cannot be better, it must be the best_score
                                if self.verbosity >= 2:
                                    print('α cutoff')
                                break # parent alpha cutoff
                    self.minimizer_score_history[str_B] = {'alpha': alpha, 'beta': beta, 'best_score': best_score}
                return self.minimizer_score_history[str_B]['best_score']
            else:
                best_score = float('+inf')
                available_moves = self.find_available_moves(B = B)
                available_moves = random.sample(available_moves, len(available_moves))
                for square in available_moves:
                    B[square] = 'O'
                    score = self.minimax_score(B = B, alpha = alpha, beta = beta, isMaximizing = True)
                    B[square] = ' '
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                    if self.use_alpha_beta_pruning and beta <= alpha:
                        if self.verbosity >= 2:
                            print('α cutoff')
                        break # parent alpha cutoff
                return best_score

    def minimax_vs_minimax(self):
        """
        board_dims = (r, c)
        """
        B = {}
        for r in range(1, self.board_dims[0]+1):
            for c in range(1, self.board_dims[1]+1):
                B[(r,c)] = ' '
        if self.verbosity >= 2:
            self.printboard(B = B)
        if self.use_alternating_starting_turn:
            if random.randint(1,2) == 1:
                turn = 'X'
            else:
                turn = 'O'
        else:
            turn = 'X'
        while self.checkwin(B) is None:
            str_B = str(B)
            if turn == 'X':
                if self.use_hashmap:
                    if str_B not in self.X_all_scores_history:
                        X_all_scores = {}
                        X_best_score = float('-inf')
                        for square in self.find_available_moves(B):
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
                    available_moves = self.find_available_moves(B)
                    available_moves = random.sample(available_moves, len(available_moves))
                    for square in available_moves:
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
                        for square in self.find_available_moves(B):
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
                    available_moves = self.find_available_moves(B)
                    available_moves = random.sample(available_moves, len(available_moves))
                    for square in available_moves:
                        B[square] = 'O'
                        score = self.minimax_score(B = B, isMaximizing = True) # see what score 'O' can still have in this situation, assuming 'X' plays optimally next
                        B[square] = ' '
                        if score < O_best_score:
                            O_best_score = score       
                            omove = square
                B[omove] = 'O'
                turn = 'X'
            if self.verbosity >= 1:
                self.printboard(B = B)
        return self.checkwin(B = B)

    def trials(self, n_trials = 10000, verbosity = 0, board_dims = (3, 3), use_hashmap = True, use_alpha_beta_pruning = True, test_early_draw = False, use_alternating_starting_turn = False, track_memory_usage = False):
        """
        board_dims = (r, c)
        """
        self.use_hashmap = use_hashmap
        self.use_alpha_beta_pruning = use_alpha_beta_pruning
        self.test_early_draw = test_early_draw # when horizontal, vertical, diagnoal has OX, thus impossible to win (not much help with memory though)
        self.use_alternating_starting_turn = use_alternating_starting_turn # if True, it randomly choose whether it starts out with 'X' or 'O'; otherwise, always 'X' 
        self.verbosity = verbosity # 0: nothing, 1: printboard, 2: printboard + alpha-beta pruning msg
        #
        if board_dims[0] != board_dims[1]:
            print('Error: it must be a square board')
            exit(1)
        else:
            self.board_dims = board_dims
        x_won, o_won, draw = 0, 0, 0
        if track_memory_usage:
            tracemalloc.start()
            self.snapshot1 = tracemalloc.take_snapshot()
            stats1 = self.snapshot1.statistics('lineno') # https://docs.python.org/3/library/tracemalloc.html
            memory_allocated1 = sum(stat.size for stat in stats1)
        self.start = time.time()
        for i in range(1, n_trials+1):
            if self.verbosity == 1:
                print(f"--------\nGame #{i}\n")
            res = self.minimax_vs_minimax()
            if res == 1:
                x_won += 1
            elif res == -1:
                o_won += 1
            else:
                draw += 1
        self.end = time.time()
        if track_memory_usage:
            self.snapshot2 = tracemalloc.take_snapshot()
            stats2 = self.snapshot2.statistics('lineno') # https://docs.python.org/3/library/tracemalloc.html
            memory_allocated2 = sum(stat.size for stat in stats2)
            tracemalloc.stop()
            add_info = f", incremental memory allocated: {(memory_allocated2 - memory_allocated1):,d} KiB"
        else:
            add_info = ""
        print(f"board_dims = {self.board_dims}, X won #: {x_won}, O won #: {o_won}, draw #: {draw:,d}, elapsed time: {self.end - self.start:.3f} sec{add_info}")
