def solve():
    n = int(input())

    def is_palindrome(s):
        return s == s[::-1]

    def evaluate(s):
        try:
            return eval(s)
        except:
            return -1

    def find_palindrome(length):
        if length == 0:
            return

        for first_digit in "123456789":
            if length == 1:
                if int(first_digit) == n:
                    print(first_digit)
                    return True
                continue

            # Palindromes of odd length
            if length % 2 == 1:
                mid_index = length // 2
                for middle_element in "123456789*":
                    s_list = [None] * length
                    s_list[mid_index] = middle_element
                    s_list[0] = first_digit
                    s_list[-1] = first_digit

                    def generate_odd_palindromes(index):
                        if index > mid_index -1 :
                            s = "".join(s_list)
                            if s[0].isdigit() and is_palindrome(s) and evaluate(s) == n:
                                print(s)
                                return True
                            return False

                        for char in "123456789*":
                            s_list[index] = char
                            s_list[length - 1 - index] = char
                            if generate_odd_palindromes(index + 1):
                                return True
                        return False

                    if generate_odd_palindromes(1):
                        return True

            # Palindromes of even length
            else:
                def generate_even_palindromes(index):
                    if index >= length // 2:
                        s = "".join(s_list)
                        if s[0].isdigit() and is_palindrome(s) and evaluate(s) == n:
                            print(s)
                            return True
                        return False

                    for char in "123456789*":
                        s_list[index] = char
                        s_list[length - 1 - index] = char
                        if generate_even_palindromes(index + 1):
                            return True
                    return False

                s_list = [None] * length
                s_list[0] = first_digit
                s_list[-1] = first_digit
                if generate_even_palindromes(1):
                    return True
        return False

    for length in range(1, 1001):
        if find_palindrome(length):
            return

    print("-1")

solve()