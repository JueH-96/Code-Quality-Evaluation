def solve():
    a, b, c = map(int, input().split())
    total_sum = a + b + c
    
    # Check for division into two groups
    if total_sum % 2 == 0:
        half_sum = total_sum / 2
        if a == half_sum or b == half_sum or c == half_sum or a + b == half_sum or a + c == half_sum or b + c == half_sum:
            print("Yes")
            return
            
    # Check for division into three groups
    if a == b and b == c:
        print("Yes")
        return
        
    print("No")

if __name__ == '__main__':
    solve()