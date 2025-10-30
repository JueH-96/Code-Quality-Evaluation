import sys
input = sys.stdin.readline

def main(): 
    N = int(input())
    S = input().strip()
    max_len = 0
    if S == "/": 
        max_len = 1
    else:
        cnt_1 = cnt_2 = 0
        for i in range(N): # Traverse from right to left until a / is encountered
            if i == 0: 
                if S[i] != "/": 
                    break
                else:
                    max_len = 1
            if S[i] == "1": 
                cnt_1 += 1
            elif S[i] == "2": 
                cnt_2 += 1
            else: 
                if cnt_1 >= cnt_2: 
                    if 2 * cnt_2 + 1 == i + 1: 
                        max_len = max(max_len, 2 * cnt_2 + 1)
                    elif 2 * (cnt_2 + 1) == i + 1: 
                        max_len = max(max_len, 2 * (cnt_2 + 1))
                    else: 
                        break
                cnt_2, cnt_1 = 0, 0
                max_len = max(max_len, i - (i + 1) // 2 + 1) # Calculate the length when a / is encountered
            
        cnt_1 = cnt_2 = 0
        for i in range(N): # Traverse from left to right
            if S[i] == "1": 
                cnt_1 += 1
            elif S[i] == "2": 
                cnt_2 += 1
            else: 
                if cnt_1 >= cnt_2: 
                    if 2 * cnt_2 + 1 == i + 1: 
                        max_len = max(max_len, 2 * cnt_2 + 1)
                    elif 2 * (cnt_2 + 1) == i + 1: 
                        max_len = max(max_len, 2 * (cnt_2 + 1))
                
                cnt_2, cnt_1 = 0, 0
                    
    print(max_len)

if __name__ == '__main__':
    main()