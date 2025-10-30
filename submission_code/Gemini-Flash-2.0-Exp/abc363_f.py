def solve():
    n = int(input())

    def evaluate(s):
        try:
            return eval(s)
        except:
            return -1

    def is_palindrome(s):
        return s == s[::-1]

    def find_solution(length):
        if length > 10:
            return None
        
        if length == 1:
            if 1 <= n <= 9:
                return str(n)
            else:
                return None

        for i in range(10**(length-1), 10**length):
            s = str(i)
            if '0' in s:
                continue
            
            if is_palindrome(s):
                if evaluate(s) == n:
                    return s
        
        return None

    def find_solution_with_multiplication(length):
        if length > 10:
            return None
        
        def backtrack(index, current_string):
            if index == length:
                if is_palindrome(current_string) and current_string[0].isdigit():
                    val = evaluate(current_string)
                    if val == n:
                        return current_string
                    else:
                        return None
                else:
                    return None
            
            for digit in "123456789":
                result = backtrack(index + 1, current_string + digit)
                if result:
                    return result
            
            if index > 0 and current_string[-1].isdigit():
                result = backtrack(index + 1, current_string + "*")
                if result:
                    return result
            
            return None

        for l in range(1, length + 1):
            result = backtrack(0, "")
            if result:
                return result
        return None

    for length in range(1, 11):
        solution = find_solution(length)
        if solution:
            print(solution)
            return

    for length in range(2, 11):
        solution = find_solution_with_multiplication(length)
        if solution:
            print(solution)
            return
    
    print("-1")

solve()