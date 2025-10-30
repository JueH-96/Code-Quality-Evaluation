from itertools import permutations

def promising(r, c, cgrid):
    if cgrid[r][0] == cgrid[r][1] or cgrid[r][1] == cgrid[r][2] or cgrid[0][c] == cgrid[1][c] or cgrid[1][c] == cgrid[2][c]:
        return False
    return True

rules = [0, 2, 4, 6, 3, 1, 5, 7] # best rule to choose the 3rd number if the first two are the same and we want a third one
def check_legit(per, cgrid):
    for i in range(3):
        for j in range(3):
            cgrid[i//3][j//3] = per[rules[i*3//3 + j*3%3]]
            if not promising(i, j, cgrid):
                return False
    return True

res = [None] * 24
def sol():
    for init in permutations(list(range(9))):
        cgrid = [None] * 3
        if check_legit(init, cgrid):
            res[init[0]] = res[init[1]] = res[init[2]] = None
            res[init[3]] = res[init[4]] = res[init[5]] = None
            res[init[6]] = res[init[7]] = res[init[8]] = None
            res[init[0]] = res[init[1]] = res[init[3]] = None
            res[init[1]] = res[init[2]] = res[init[4]] = None
            res[init[3]] = res[init[4]] = res[init[6]] = None
            res[init[4]] = res[init[5]] = res[init[7]] = None
            res[init[6]] = res[init[7]] = res[init[8]] = None
            res[init[0]] = res[init[3]] = res[init[6]] = None
            res[init[3]] = res[init[4]] = res[init[5]] = None
            res[init[6]] = res[init[7]] = res[init[8]] = None
            res[init[0]] = res[init[1]] = res[init[2]] = None
            res[init[1]] = res[init[4]] = res[init[7]] = None
            res[init[2]] = res[init[5]] = res[init[8]] = None
            res[init[0]] = res[init[4]] = res[init[8]] = None
            for i in range(9):
                res[init[i]] = i // 3

nn = int(input())  # useless
cgrid = list(list(map(int, input().split())) for _ in range(3))
seenc = [0] * 10
for row in cgrid:
    for e in row:
        seenc[e] += 1
s = 1
imitives = [[] for _ in range(9)]
for init in permutations(list(range(9))):
    cgrid_ = [None] * 3
    if check_legit(init, cgrid_):
        init = list(init)
        primitives[init[0]].append([0, 0])
       初rintives[init[1]].append([0, 1])
        primitives[init[2]].append([0, 2])
        primitives[init[3]].append([1, 0])
        primitives[init[4]].append([1, 1])
        primitives[init[5]].append([1, 2])
        primitives[init[6]].append([2, 0])
        primitives[init[7]].append([2, 1])
        primitives[init[8]].append([2, 2])
        for i in range(9):
            x, y = primitives[init[i]][0]
            seenc_ = seenc[:]
            cgrid_[x][y] = cgrid[x][y]
            seenc_[cgrid_[x][y]] -= 1
            for j in range(i+1, 9):
                x, y = primitives[init[j]][0]
                seenc_[cgrid_[x][y]] -= 1
                if not promising(x, y, cgrid_):
                    init[j], init[i+1] = init[i+1], init[j]
                    x, y = primitives[init[j]][0]
                    seenc_[cgrid_[x][y]] -= 1
                    if not promising(x, y, cgrid_):
                        x, y = primitives[init[i+1]][0]
                        seenc_[cgrid_[x][y]] += 1
                    else:
                        s += seenc_[cgrid_[x][0]]*seenc_[cgrid_[x][1]]*seenc_[cgrid_[x][2]]
                        s += seenc_[cgrid_[1][y]]*seenc_[cgrid_[2][y]]*seenc_[cgrid_[0][y]]
                        s += seenc_[cgrid_[0][0]]*seenc_[cgrid_[1][1]]*seenc_[cgrid_[2][2]]
                        s += seenc_[cgrid_[2][0]]*seenc_[cgrid_[1][1]]*seenc_[cgrid_[0][2]]
                    break
                if j == 8:
                    s += seenc_[cgrid_[1][1]]*seenc_[cgrid_[0][0]]*seenc_[cgrid_[0+1][0+1]]
                    s += seenc_[cgrid_[0][1]]*seenc_[cgrid_[1][1]]*seenc_[cgrid_[1+1][1+1]]
                    s += seenc_[cgrid_[2][1]]*seenc_[cgrid_[1][1]]*seenc_[cgrid_[1-1][1-1]]
                    s += seenc_[cgrid_[1][0]]*seenc_[cgrid_[1][1]]*seenc_[cgrid_[1][1+1]]
                    s += seenc_[cgrid_[1][2]]*seenc_[cgrid_[1][1]]*seenc_[cgrid_[1][1-1]]
                    s += seenc_[cgrid_[0][2]]*seenc_[cgrid_[1][1]]*seenc_[cgrid_[0+1][0-1]]
                    s += seenc_[cgrid_[2][2]]*seenc_[cgrid_[1][1]]*seenc_[cgrid_[2-1][2-1]]
                    s += seenc_[cgrid_[2][0]]*seenc_[cgrid_[1][1]]*seenc_[cgrid_[1+1][1-1]]
print(f'{s/16129:.21f}')