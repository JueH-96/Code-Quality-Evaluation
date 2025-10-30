import math

def main():
    n = int(input().strip())
    
    m = 0
    if n > 1:
        m = math.ceil(math.log2(n))
    else:
        m = 1
        
    print(m)
    
    friends = [[] for _ in range(m)]
    
    for bottle in range(1, n + 1):
        num = bottle - 1
        for i in range(m):
            if (num >> i) & 1:
                friends[i].append(bottle)
                
    for lst in friends:
        k = len(lst)
        if k == 0:
            print("0")
        else:
            sorted_list = sorted(lst)
            line = str(k) + " " + " ".join(map(str, sorted_list))
            print(line)
            
    s = input().strip()
    
    num_val = 0
    for idx, char in enumerate(s):
        if char == '1':
            num_val |= (1 << idx)
            
    spoiled_bottle = num_val + 1
    print(spoiled_bottle)

if __name__ == "__main__":
    main()