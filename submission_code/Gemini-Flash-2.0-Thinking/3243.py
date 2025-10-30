class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def countPowerfulUpTo(n: int, limit: int, s: str) -> int:
            n_str = str(n)
            len_n = len(n_str)
            len_s = len(s)
            count = 0

            def is_powerful(num_str):
                if not num_str.endswith(s):
                    return False
                for digit in num_str:
                    if int(digit) > limit:
                        return False
                return True

            # Optimization: Directly construct powerful numbers

            def generate_powerful(length):
                if length == len_s:
                    if all(int(d) <= limit for d in s):
                        yield s
                    return

                if length > len_s:
                    prefix_len = length - len_s
                    for digits in self.product(range(limit + 1), repeat=prefix_len):
                        prefix = "".join(map(str, digits))
                        yield prefix + s

            for length in range(1, len_n + 1):
                for powerful_str in generate_powerful(length):
                    if len(powerful_str) == length:
                        powerful_int = int(powerful_str)
                        if powerful_int <= n:
                            count += 1

            return count

        def countPowerfulUpTo_optimized(n: int, limit: int, s: str) -> int:
            n_str = str(n)
            len_n = len(n_str)
            len_s = len(s)
            count = 0

            def check(num_str):
                return num_str.endswith(s) and all(int(d) <= limit for d in num_str)

            # Powerful numbers of length len_s
            if len_s <= len_n and int(s) <= n and all(int(d) <= limit for d in s):
                count += 1

            # Powerful numbers of length > len_s
            for length in range(len_s + 1, len_n + 1):
                prefix_len = length - len_s

                def dp(index, is_tight):
                    if index == prefix_len:
                        num_str = "".join(current_prefix) + s
                        return 1

                    if (index, is_tight) in memo:
                        return memo[(index, is_tight)]

                    upper = int(n_str[index]) if is_tight else limit
                    res = 0
                    for digit in range(upper + 1):
                        current_prefix.append(str(digit))
                        res += dp(index + 1, is_tight and (digit == upper))
                        current_prefix.pop()
                    memo[(index, is_tight)] = res
                    return res

                memo = {}
                current_prefix = []
                if prefix_len <= len_n:
                    count += dp(len_n - length, True)

            return count

        def countPowerfulUpTo_v2(n: int, limit: int, s: str) -> int:
            count = 0
            for length in range(1, len(str(n)) + 2):
                if length < len(s):
                    continue

                if length == len(s):
                    if all(int(d) <= limit for d in s) and int(s) <= n:
                        count += 1
                    continue

                prefix_len = length - len(s)

                def generate(idx, current_num_str):
                    nonlocal count
                    if idx == prefix_len:
                        potential_powerful = current_num_str + s
                        if len(potential_powerful) == length and int(potential_powerful) <= n:
                            count += 1
                        return

                    upper_bound = limit
                    for digit in range(upper_bound + 1):
                        generate(idx + 1, current_num_str + str(digit))

                generate(0, "")
            return count

        def countPowerful(n, limit, s):
            n_str = str(n)
            len_n = len(n_str)
            len_s = len(s)
            count = 0

            for length in range(len_s, len_n + 1):
                if length == len_s:
                    if all(int(d) <= limit for d in s) and int(s) <= n:
                        count += 1
                elif length > len_s:
                    prefix_len = length - len_s
                    max_prefix = int(n_str[:prefix_len]) if length <= len_n else -1

                    def generate_prefixes(index, current_prefix_val):
                        c = 0
                        if index == prefix_len:
                            powerful_num = int(str(current_prefix_val) + s)
                            return 1 if powerful_num <= n else 0

                        upper = limit
                        if length <= len_n and index < prefix_len -1 :
                            pass
                        elif length <= len_n and index == prefix_len - 1:
                            upper = int(n_str[index])

                        for digit in range(upper + 1):
                            c += generate_prefixes(index + 1, current_prefix_val * 10 + digit)
                        return c

                    # Simplified counting
                    max_prefix_str = n_str[:prefix_len]

                    def dp(index, is_tight):
                        if index == prefix_len:
                            return 1

                        if (index, is_tight) in memo:
                            return memo[(index, is_tight)]

                        upper = int(max_prefix_str[index]) if is_tight else limit
                        ans = 0
                        for digit in range(upper + 1):
                            if digit <= limit:
                                ans += dp(index + 1, is_tight and (digit == upper))
                        memo[(index, is_tight)] = ans
                        return ans

                    memo = {}
                    count += dp(0, True)

            return count

        def countPowerfulUp(n, limit, s):
            n_str = str(n)
            len_n = len(n_str)
            len_s = len(s)
            count = 0

            for length in range(len_s, len_n + 1):
                if length == len_s:
                    if all(int(d) <= limit for d in s) and int(s) <= n:
                        count += 1
                elif length > len_s:
                    prefix_len = length - len_s

                    def dp(index, is_tight):
                        if index == prefix_len:
                            return 1
                        if (index, is_tight) in memo:
                            return memo[(index, is_tight)]

                        upper = int(n_str[index]) if is_tight else limit
                        ans = 0
                        for digit in range(upper + 1):
                            if digit <= limit:
                                ans += dp(index + 1, is_tight and (digit == upper))
                        memo[(index, is_tight)] = ans
                        return ans

                    memo = {}
                    count += dp(0, True)
            return count

        def countPowerfulUpFinal(n, limit, s):
            s_int = int(s)
            s_len = len(s)
            n_str = str(n)
            n_len = len(n_str)
            count = 0

            for length in range(s_len, n_len + 1):
                if length == s_len:
                    if all(int(d) <= limit for d in s) and s_int <= n:
                        count += 1
                else:
                    prefix_len = length - s_len
                    n_prefix_str = n_str[:prefix_len]

                    def dp(index, is_tight):
                        if index == prefix_len:
                            return 1
                        if (index, is_tight) in memo:
                            return memo[(index, is_tight)]

                        upper = int(n_prefix_str[index]) if is_tight else limit
                        ans = 0
                        for digit in range(upper + 1):
                            if digit <= limit:
                                ans += dp(index + 1, is_tight and (digit == upper))
                        memo[(index, is_tight)] = ans
                        return ans

                    memo = {}
                    count += dp(0, True)
            return count

        return countPowerfulUpFinal(finish, limit, s) - countPowerfulUpFinal(start - 1, limit, s)