from collections import deque, defaultdict

class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # Define the string properties: (name, start_char, start_count, end_char, end_count, uses)
        string_info = [
            ('AA', 'A', 2, 'A', 2),
            ('BB', 'B', 2, 'B', 2),
            ('AB', 'A', 1, 'B', 1)
        ]

        # The DP will be a dictionary where the key is a tuple (a, b, c, last_char, end_count)
        # and the value is the maximum length achieved in that state.
        dp = defaultdict(int)
        initial_state = (0, 0, 0, None, 0)
        dp[initial_state] = 0

        queue = deque()
        queue.append(initial_state)

        max_length = 0

        while queue:
            current_state = queue.popleft()
            a, b, c, last_char, end_count = current_state
            current_length = dp[current_state]

            # If this state's length is less than the current max, skip
            if current_length < max_length:
                continue

            # Try adding each possible string
            # "AA"
            if a < x:
                # String info: start_char 'A', start_count 2
                if last_char == 'A':
                    if end_count + 2 >= 3:
                        # Transition invalid
                        pass
                    else:
                        # Transition valid
                        new_a = a + 1
                        new_b = b
                        new_c = c
                        new_last_char = 'A'
                        new_end_count = 2
                        new_length = current_length + 2
                        new_state = (new_a, new_b, new_c, new_last_char, new_end_count)
                        if new_state not in dp or dp[new_state] < new_length:
                            dp[new_state] = new_length
                            if new_length > max_length:
                                max_length = new_length
                            queue.append(new_state)
                else:
                    # Transition valid
                    new_a = a + 1
                    new_b = b
                    new_c = c
                    new_last_char = 'A'
                    new_end_count = 2
                    new_length = current_length + 2
                    new_state = (new_a, new_b, new_c, new_last_char, new_end_count)
                    if new_state not in dp or dp[new_state] < new_length:
                        dp[new_state] = new_length
                        if new_length > max_length:
                            max_length = new_length
                        queue.append(new_state)

            # "BB"
            if b < y:
                # String info: start_char 'B', start_count 2
                if last_char == 'B':
                    if end_count + 2 >= 3:
                        # Transition invalid
                        pass
                    else:
                        # Transition valid
                        new_a = a
                        new_b = b + 1
                        new_c = c
                        new_last_char = 'B'
                        new_end_count = 2
                        new_length = current_length + 2
                        new_state = (new_a, new_b, new_c, new_last_char, new_end_count)
                        if new_state not in dp or dp[new_state] < new_length:
                            dp[new_state] = new_length
                            if new_length > max_length:
                                max_length = new_length
                            queue.append(new_state)
                else:
                    # Transition valid
                    new_a = a
                    new_b = b + 1
                    new_c = c
                    new_last_char = 'B'
                    new_end_count = 2
                    new_length = current_length + 2
                    new_state = (new_a, new_b, new_c, new_last_char, new_end_count)
                    if new_state not in dp or dp[new_state] < new_length:
                        dp[new_state] = new_length
                        if new_length > max_length:
                            max_length = new_length
                        queue.append(new_state)

            # "AB"
            if c < z:
                # String info: start_char 'A', start_count 1
                if last_char == 'A':
                    if end_count + 1 >= 3:
                        # Transition invalid
                        pass
                    else:
                        # Transition valid
                        new_a = a
                        new_b = b
                        new_c = c + 1
                        new_last_char = 'B'
                        new_end_count = 1
                        new_length = current_length + 2
                        new_state = (new_a, new_b, new_c, new_last_char, new_end_count)
                        if new_state not in dp or dp[new_state] < new_length:
                            dp[new_state] = new_length
                            if new_length > max_length:
                                max_length = new_length
                            queue.append(new_state)
                else:
                    # Transition valid
                    new_a = a
                    new_b = b
                    new_c = c + 1
                    new_last_char = 'B'
                    new_end_count = 1
                    new_length = current_length + 2
                    new_state = (new_a, new_b, new_c, new_last_char, new_end_count)
                    if new_state not in dp or dp[new_state] < new_length:
                        dp[new_state] = new_length
                        if new_length > max_length:
                            max_length = new_length
                        queue.append(new_state)

        return max_length