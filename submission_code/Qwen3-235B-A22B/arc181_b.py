import sys

def main():
    input = sys.stdin.read
    data = input().split()
    t = int(data[0])
    index = 1
    results = []
    
    for _ in range(t):
        S = data[index]
        index += 1
        X = data[index]
        index += 1
        Y = data[index]
        index += 1
        
        if X == Y:
            results.append("Yes")
            continue
        
        a = X.count('1')
        b = Y.count('1')
        X_len = len(X)
        Y_len = len(Y)
        
        if a == b:
            if X_len == Y_len:
                results.append("Yes")
            else:
                results.append("No")
            continue
        
        denominator = b - a
        len_S = len(S)
        numerator = len_S * (X_len - Y_len)
        
        if denominator == 0:
            results.append("No")
            continue
        
        if (numerator % denominator) != 0:
            results.append("No")
            continue
        
        len_T = numerator // denominator + len_S
        if len_T < 0:
            results.append("No")
            continue
        
        # Now, we need to simulate the expansion comparison
        # Build X_blocks and Y_blocks
        X_blocks = []
        for c in X:
            if c == '0':
                X_blocks.append((0, len_S))  # 0 for S
            else:
                X_blocks.append((1, len_T))  # 1 for T
        
        Y_blocks = []
        for c in Y:
            if c == '0':
                Y_blocks.append((0, len_S))
            else:
                Y_blocks.append((1, len_T))
        
        i = j = 0
        pos_x = pos_y = 0
        valid = True
        
        while i < len(X_blocks) and j < len(Y_blocks):
            type_x, size_x = X_blocks[i]
            type_y, size_y = Y_blocks[j]
            
            remaining_x = size_x - pos_x
            remaining_y = size_y - pos_y
            copy_len = min(remaining_x, remaining_y)
            
            if type_x == 0 and type_y == 0:
                # Both S blocks, compare S[pos_x : pos_x+copy_len] == S[pos_y : pos_y+copy_len]
                for k in range(copy_len):
                    if S[pos_x + k] != S[pos_y + k]:
                        valid = False
                        break
                if not valid:
                    break
            elif type_x == 0 and type_y == 1:
                # S vs T: T needs to be S[pos_x + k] at pos_y + k
                # T is unknown, so we can't check, assume valid for now
                pass
            elif type_x == 1 and type_y == 0:
                # T vs S: T needs to be S[pos_y + k] at pos_x + k
                pass
            else:
                # Both T blocks: T[pos_x + k] == T[pos_y + k], but T is unknown
                pass
            
            # Advance pointers
            pos_x += copy_len
            pos_y += copy_len
            
            if pos_x == size_x:
                i += 1
                pos_x = 0
            if pos_y == size_y:
                j += 1
                pos_y = 0
        
        if valid:
            results.append("Yes")
        else:
            results.append("No")
    
    sys.stdout.write("
".join(results) + "
")

if __name__ == "__main__":
    main()