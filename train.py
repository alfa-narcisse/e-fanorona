import pygame, sys, math, copy
import numpy as np

pygame.init()

WIDTH, HEIGHT = 600, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fanorona-3 avec IA")

WHITE = (255,255,255)
BLACK = (0,0,0)
RED   = (200, 60, 60)    # Joueur humain
BLUE  = (60, 60, 200)    # IA
BG    = (230,230,230)
MOVED_AI = (153,0,76)
MOVED_P = (0,0,102)
RADIUS = 18

# Les 9 nœuds du plateau 3×3
NODES = [
    (100,100), (300,100), (500,100),
    (100,300), (300,300), (500,300),
    (100,500), (300,500), (500,500)
]

# Les connexions (plateau en X)
EDGES = [
    (0,1),(1,2),(3,4),(4,5),(6,7),(7,8),
    (0,3),(3,6),(1,4),(4,7),(2,5),(5,8),
    (0,4),(2,4),(6,4),(8,4)
]

LINES = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6)
]

# --- Minimax avec alpha-beta ---
# ---------- Configuration ----------

initial_side_IA = [0,1,2]
initial_side_adv = [6,7,8]

plateau_Game = [
    1,1,1,
    0,0,0,
    -1,-1,-1
]

MINIMAX_DEPTH = 7
# ajustable (profondeur de recherche)


def chek_wins(plateau:list, val_player):
    for val in LINES:
        if all([plateau[i] == val_player for i in val]):
            return True
    return False


# ---------- Utils ----------
def get_possible_moves(node, plateau):
    moves = []
    for u,v in EDGES:
        if node == u and plateau[v] == 0:
            moves.append((u,v))
        elif node == v and plateau[u] == 0:
            moves.append((v,u))
    return moves


# ---------- EVALUATE ----------
def evaluate(plateau):
    score = 0
    for a,b,c in LINES:
        vals = [plateau[a], plateau[b], plateau[c]]
        if vals == [2, 2, 2]:
            return float('inf')
        if vals == [-2, -2, -2]:
            return -float('inf')
        if sum([v == 2 for v in vals]) >= 2:
            score += 10
        elif sum([v == -2 for v in vals]) >= 2:
            score -= 10
    for player in plateau:
        if player == 2:
            score+= 2
        elif player == -2:
            score -= 2
    return score


# ---------- Minimax alpha-beta ----------
def minimax_alpha_beta(plateau, ia_turn=True, depth=MINIMAX_DEPTH):
    val = evaluate(plateau)
    d = depth
    if math.isinf(val) or depth == 0:
        return None, val, d

    player_vals = (1,2) if ia_turn else (-1,-2)
    best_edge = None
    min_eval = math.inf
    max_eval = -math.inf
    dico = {}
    for node in range(9):
        if plateau[node] in player_vals:
            move = get_possible_moves(node, plateau)
            depthRefInfmin = MINIMAX_DEPTH
            dephtRefMax = 0
            for  edge_index, target_node in move:
                new_plateau = plateau.copy()
                new_plateau[target_node] = player_vals[1]
                new_plateau[node] = 0
                _, score, d = minimax_alpha_beta(
                    new_plateau, ia_turn= not ia_turn,
                    depth=depth-1)
                if ia_turn:
                    if score >= max_eval:
                        val = score
                        max_eval = score
                        best_edge = (edge_index, target_node)
                        if score == -math.inf or score==math.inf:
                            dico[best_edge] = d
                else:
                    if score <= min_eval:
                        val = score
                        min_eval = score
                        best_edge = (edge_index,target_node)

    if ia_turn and val== math.inf:
        #On cherche le best_target dont la profondeur est plus grand(plus proche)/ en cas d'égalité, on prend le premier
        d_max = 0
        b_move = tuple()
        for k,v in dico.items():
            if dico[k]>d_max:
                b_move = k
                d_max = v
        return  b_move,val,d_max
    if ia_turn and val == -math.inf:
        #On cherche le best_target dont la profondeur est plus gpeti(plus loin)/ en cas d'égalité, on prend le premier
        d_max = MINIMAX_DEPTH + 1
        b_move = tuple()
        for k,v in dico.items():
            if v < d_max:
                b_move = k
                d_max = v
        return  b_move,val,d_max
    return best_edge, val, d

def do_the_move(move):
    dep,target = move[0], move[1]
    if plateau_Game[target] ==0:
        if plateau_Game[dep] in (-1,-2):
            plateau_Game[target] = -2
        elif plateau_Game[dep] in (1,2):
            plateau_Game[target] = 2
        plateau_Game[dep] = 0

# --- Dessin ---
def draw():
    WIN.fill(BG)
    for a,b in EDGES:
        pygame.draw.line(WIN, BLACK, NODES[a], NODES[b], 3)
    for i,pos in enumerate(NODES):
        pygame.draw.circle(WIN, BLACK,pos,RADIUS,2)
    for i,pos in enumerate(NODES):
        if plateau_Game[i]==1:
            pygame.draw.circle(WIN, RED, pos, RADIUS-3)
        elif plateau_Game[i] == 2:# L'IA en rouge/ mon adversaire
            pygame.draw.circle(WIN, MOVED_AI, pos, RADIUS - 3)
        elif plateau_Game[i] == -1:
            pygame.draw.circle(WIN, BLUE,pos,RADIUS-3) # Je suis en bleu
        elif plateau_Game[i] == -2:
            pygame.draw.circle(WIN, MOVED_P,pos,RADIUS-3) # Je suis en bleu
    pygame.display.update()


def node_at_pos(mouse_pos):
    for i,(x,y) in enumerate(NODES):
        if math.dist(mouse_pos,(x,y))<RADIUS+5:
            return i
    return None


# ---------------- MAIN LOOP ----------------
selected = None
IA_turn = False
while True:
    draw()
    # Tour IA
    if IA_turn:
        pygame.time.wait(400)
        best = None
        while best is None:
            best,_,_ = minimax_alpha_beta(plateau_Game)
        do_the_move(best)
        if chek_wins(plateau_Game, 2):
            draw()
            print("Tu as perdu :(")
            pygame.time.wait(3000)
            pygame.quit(); sys.exit()

        IA_turn = not IA_turn
        continue

    # Tour humain
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit(); sys.exit()

        if event.type==pygame.MOUSEBUTTONDOWN:
            node = node_at_pos(event.pos)
            if node is None: continue

            if selected is None:
                if plateau_Game[node] in (-1,-2):
                    selected=node
            else:
                if (selected, node) in get_possible_moves(selected, plateau_Game):
                    do_the_move((selected,node))
                    selected=None
                    if chek_wins(plateau_Game, -2):
                        draw()
                        print("Tu as gagné !")
                        pygame.time.wait(2000)
                        pygame.quit(); sys.exit()

                    IA_turn = not IA_turn
                else:
                    if plateau_Game[node] in (-1,-2):
                        selected=node
                    else:
                        selected=None
