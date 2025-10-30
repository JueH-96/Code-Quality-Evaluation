def solve():
    k = int(input())
    
    nums = []
    
    def generate(current_num, last_digit):
        nums.append(current_num)
        
        for i in range(last_digit):
            generate(current_num * 10 + i, i)
            
    for i in range(1, 10):
        generate(i, i)
        
    nums.sort()
    
    print(nums[k-1])

solve()