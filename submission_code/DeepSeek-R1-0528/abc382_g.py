import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        K = int(data[index])
        S_x = int(data[index+1])
        S_y = int(data[index+2])
        T_x = int(data[index+3])
        T_y = int(data[index+4])
        index += 5
        
        twoK = 2 * K
        i_s = (2 * S_x + 1) // twoK
        j_s = (2 * S_y + 1) // twoK
        i_t = (2 * T_x + 1) // twoK
        j_t = (2 * T_y + 1) // twoK
        
        between = abs(i_s - i_t) + abs(j_s - j_t)
        
        if (i_s + j_s) % 2 == 0:
            k_s = (2 * S_y + 1 - 2 * j_s * K) // 2
        else:
            k_s = (2 * S_x + 1 - 2 * i_s * K) // 2
        
        extra = 0
        if (i_s + j_s) % 2 == 1:
            if k_s != 0 and k_s != K - 1:
                extra = 1
        
        ans = between + extra
        results.append(str(ans))
    
    print("
".join(results))

if __name__ == "__main__":
    main()