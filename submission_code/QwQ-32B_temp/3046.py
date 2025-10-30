class Solution:
    def minimumOperations(self, num: str) -> int:
        def get_min_for_pair(X, Y):
            min_steps = float('inf')
            y_positions = []
            # Collect all positions of Y from the end
            for i in reversed(range(len(num))):
                if num[i] == Y:
                    y_positions.append(i)
            # Check each Y position to find the best X before it
            for y in y_positions:
                x_pos = -1
                # Look for X in reverse from y-1 to 0
                for j in reversed(range(y)):
                    if num[j] == X:
                        x_pos = j
                        break
                if x_pos != -1:
                    steps = (len(num) - x_pos - 2)
                    if steps < min_steps:
                        min_steps = steps
            return min_steps if min_steps != float('inf') else float('inf')

        # Evaluate all possible pairs
        pairs = [("0", "0"), ("2", "5"), ("5", "0"), ("7", "5")]
        min_pair = float('inf')
        for X, Y in pairs:
            current = get_min_for_pair(X, Y)
            if current < min_pair:
                min_pair = current

        # Evaluate the zero case
        has_zero = '0' in num
        option1 = (len(num) - 1) if has_zero else float('inf')
        option2 = len(num)  # delete all to get 0
        zero_case = min(option1, option2)

        # The answer is the minimum between the best pair and zero_case
        ans = min(min_pair, zero_case)
        return ans if ans != float('inf') else len(num)  # in case all are inf, but deleting all is possible