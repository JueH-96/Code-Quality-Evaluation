def solve():
    s = input()
    q = int(input())
    queries = list(map(int, input().split()))
    results = []
    initial_s = list(s)
    initial_length = len(initial_s)
    
    def invert_case(char):
        if 'a' <= char <= 'z':
            return char.upper()
        elif 'A' <= char <= 'Z':
            return char.lower()
        return char
        
    def get_char_at_k(k):
        n = 60 # Number of operations, can be set to a sufficiently large value like 60
        current_k = k
        invert_flag = False
        current_s = initial_s
        current_length = initial_length
        
        while n > 0:
            prev_length = current_length // 2
            if current_k <= prev_length:
                current_length = prev_length
                n -= 1
            else:
                current_k -= prev_length
                current_length = prev_length
                invert_flag = not invert_flag
                n -= 1
                
        char_index = current_k - 1
        original_char = initial_s[char_index]
        if invert_flag:
            return invert_case(original_char)
        else:
            return original_char
            
    for k in queries:
        results.append(get_char_at_k(k))
        
    print(*(results))

if __name__ == '__main__':
    solve()