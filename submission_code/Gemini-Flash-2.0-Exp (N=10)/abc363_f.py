def solve():
    n = int(input())

    def is_palindrome(s):
        return s == s[::-1]

    def evaluate(s):
        try:
            return eval(s)
        except:
            return -1

    def find_palindrome(target, current_str="", length=0):
        if length > 1000:
            return None
        
        if current_str and is_palindrome(current_str) and current_str[0].isdigit() and evaluate(current_str) == target:
            return current_str
        
        if length > 0 and len(current_str) > 0 and not current_str[0].isdigit():
            return None
        
        if length > 0 and len(current_str) > 0 and not is_palindrome(current_str):
            return None

        if length == 0:
            for digit in "123456789":
                res = find_palindrome(target, digit, length + 1)
                if res:
                    return res
            return None
        
        for char in "123456789*":
            res = find_palindrome(target, current_str + char, length + 1)
            if res:
                return res
        return None

    result = find_palindrome(n)
    if result:
        print(result)
    else:
        print("-1")

solve()