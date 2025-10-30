def is_321_like(n):
    s = str(n)
    if len(s) == 1:
        return True
    for i in range(len(s) - 1):
        if not (int(s[i]) > int(s[i+1])):
            return False
    return True

def solve():
    k = int(input())
    count = 0
    current_number = 0
    result = -1
    
    generated_numbers = []
    
    def generate_321_like_numbers(current_num_str):
        num = 0
        if current_num_str:
            num = int(current_num_str)
            if num > 0:
                generated_numbers.append(num)
                
        if len(current_num_str) >= 10:
            return
            
        last_digit = 9
        if current_num_str:
            last_digit = int(current_num_str[-1]) - 1
            
        for next_digit in range(last_digit, -1, -1):
            if len(current_num_str) == 0 and next_digit == 0:
                continue
            generate_321_like_numbers(current_num_str + str(next_digit))
            
    generate_321_like_numbers("")
    generated_numbers.sort()
    
    print(generated_numbers[k-1])

if __name__ == '__main__':
    solve()