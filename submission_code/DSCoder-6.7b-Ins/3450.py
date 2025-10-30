class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # Initialize the child list
        children = list(range(n))
        # Initialize the direction of passing the ball
        direction = 1

        # Loop through each second
        for _ in range(k):
            # If the direction is towards the right
            if direction == 1:
                # Pass the ball to the next child
                children = children[1:] + children[:1]
            # If the direction is towards the left
            else:
                # Pass the ball to the previous child
                children = children[-1:] + children[:-1]

            # Reverse the direction of passing the ball
            direction *= -1

        # Return the child who receives the ball
        return children[0]