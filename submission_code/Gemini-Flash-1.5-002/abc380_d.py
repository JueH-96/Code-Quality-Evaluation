def solve():
    s = input()
    q = int(input())
    queries = list(map(int, input().split()))

    def transform(char):
        if 'a' <= char <= 'z':
            return char.upper()
        else:
            return char.lower()

    def get_char_at(index):
        n = len(s)
        current_len = n
        
        if index <= current_len:
            return s[index-1]

        iteration = 0
        while index > current_len:
            iteration +=1
            index -= current_len
            current_len *=2

        
        t = "".join(map(transform,s))
        full_s = s
        
        for i in range(min(iteration,100)):
            full_s += "".join(map(transform,full_s))

        return full_s[index-1]

    result = []
    for k in queries:
        result.append(get_char_at(k))

    print(*result)

solve()