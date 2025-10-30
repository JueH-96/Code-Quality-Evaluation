def solve():
    n = int(input())
    a = list(map(int, input().split()))
    m = int(input())
    b = list(map(int, input().split()))
    l = int(input())
    c = list(map(int, input().split()))
    q = int(input())
    x_queries = list(map(int, input().split()))
    
    c_set = set(c)
    results = []
    
    for target_sum in x_queries:
        found_solution = False
        for val_a in a:
            for val_b in b:
                target_c_val = target_sum - val_a - val_b
                if target_c_val in c_set:
                    found_solution = True
                    break
            if found_solution:
                break
        if found_solution:
            results.append("Yes")
        else:
            results.append("No")
            
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()