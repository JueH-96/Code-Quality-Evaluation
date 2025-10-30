# YOUR CODE HERE
def solve():
    k = int(input())
    
    ans = []
    
    def generate(current, last_digit):
        ans.append(int("".join(map(str, current))))
        
        for digit in range(last_digit - 1, -1, -1):
            generate(current + [digit], digit)
            
    for i in range(1, 10):
        generate([i], i)
        
    ans.sort()
    print(ans[k-1])

solve()