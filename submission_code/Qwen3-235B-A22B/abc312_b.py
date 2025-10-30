# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    S = []
    for _ in range(N):
        S.append(input[idx])
        idx += 1
    
    ans = []
    for i in range(N - 8):
        for j in range(M - 8):
            valid = True
            
            # Check top-left 3x3
            for x in range(3):
                if not valid:
                    break
                for y in range(3):
                    if S[i + x][j + y] != '#':
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check bottom-right 3x3
            for x in range(6, 9):
                if not valid:
                    break
                for y in range(6, 9):
                    if S[i + x][j + y] != '#':
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check top-left perimeter (4x4 square minus 3x3)
            for x in range(4):
                if not valid:
                    break
                for y in range(4):
                    if x < 3 and y < 3:
                        continue
                    if S[i + x][j + y] != '.':
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check bottom-right perimeter (4x4 square starting at 5,5 minus 3x3)
            for x in range(5, 9):
                if not valid:
                    break
                for y in range(5, 9):
                    if x >= 6 and y >= 6:
                        continue
                    if S[i + x][j + y] != '.':
                        valid = False
                        break
            
            if valid:
                ans.append((i + 1, j + 1))
    
    for a in ans:
        print(a[0], a[1])

if __name__ == '__main__':
    main()