import re, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1 — The knowledges
# This is the chatbot's brain — all the questions it knows about and the answers it can give.
knowledges = [
    {
        "patterns": [
            "what is python", "what is python used for",
            "tell me about python", "explain python",
            "why learn python", "why use python"
        ],
        "response": (
            "Python is a high-level, beginner-friendly programming language.\n"
            "It is used for:\n"
            " - Web development\n"
            " - Data Science and Machine Learning\n"
            " - Automation and scripting\n"
            " - APIs and backend services\n"
            " - Game development\n\n"
            "It is one of the most popular languages in the world because\n"
            "it reads almost like plain English!"
        ),
        "tag": "basics"
    },
    {
        "patterns": [
            "how do i print", "print in python", "print statement",
            "how to display output", "how to show text", "print hello world"
        ],
        "response": (
            "Printing in Python:\n\n"
            "print('Hello World')\n"
            "print('My name is', name)\n"
            "print(f'I am {age} years old') <- f-string (recommended)\n\n"
            "# Print multiple lines:\n"
            "print('Line 1')\n"
            "print('Line 2')\n\n"
            "# Print without newline:\n"
            "print('Hello', end=' ')\n"
            "print('World')\n"
            "# Output: Hello World"
        ),
        "tag": "basics"
    },
    {
        "patterns": [
            "what is a variable", "how to create variable",
            "how to declare variable", "variable in python",
            "how to store value", "assign variable"
        ],
        "response": (
            "Variables in Python:\n\n"
            "name = 'Alice'\n"
            "age  = 25\n"
            "gpa  = 3.9\n"
            "is_student = True\n\n"
            "# Python is dynamically typed — no need to declare type!\n"
            "x = 10\n"
            "x = 'hello'\n\n"
            "# Multiple assignment:\n"
            "a, b, c = 1, 2, 3\n\n"
            "Variable names: use lowercase with underscores (snake_case)\n"
            "✔ user_name = 'Bob'\n"
            "✘ UserName = 'Bob' <- technically works but not Pythonic"
        ),
        "tag": "basics"
    },
    {
        "patterns": [
            "data types", "types in python", "what are data types",
            "int float string bool", "python types", "type of variable"
        ],
        "response": (
            "Python Data Types:\n\n"
            "int -> whole numbers          age = 25\n"
            "float -> decimal numbers        price = 9.99\n"
            "str -> text                   name = 'Alice'\n"
            "bool -> True or False          is_active = True\n"
            "list -> ordered collection     fruits = ['apple', 'banana']\n"
            "tuple -> immutable list         coords = (10, 20)\n"
            "dict -> key-value pairs        person = {'name': 'Bob', 'age': 30}\n"
            "set -> unique items only      nums = {1, 2, 3}\n\n"
            "# Check type:\n"
            "print(type(age))     # <class 'int'>\n"
            "print(type(name))    # <class 'str'>"
        ),
        "tag": "basics"
    },
    {
        "patterns": [
            "if statement", "if else", "conditional", "how to use if",
            "if elif else", "check condition", "comparison"
        ],
        "response": (
            "If / Elif / Else in Python:\n\n"
            "age = 20\n\n"
            "if age >= 18:\n"
            "    print('Adult')\n"
            "elif age >= 13:\n"
            "    print('Teenager')\n"
            "else:\n"
            "    print('Child')\n\n"
            "# One-liner (ternary):\n"
            "label = 'Adult' if age >= 18 else 'Minor'\n\n"
            "Comparison operators:\n"
            "==  equal to          !=  not equal\n"
            ">   greater than      <   less than\n"
            ">=  greater or equal  <=  less or equal\n\n"
            "Logical operators:\n"
            "and   both must be true\n"
            "or    at least one must be true\n"
            "not   reverse the condition"
        ),
        "tag": "control_flow"
    },
    {
        "patterns": [
            "for loop", "how to loop", "iterate", "range function",
            "loop through list", "for in", "looping"
        ],
        "response": (
            "For Loops in Python:\n\n"
            "for i in range(5):\n"
            "    print(i)\n\n"
            "fruits = ['apple', 'banana', 'cherry']\n"
            "for fruit in fruits:\n"
            "    print(fruit)\n\n"
            "for i, fruit in enumerate(fruits):\n"
            "    print(i, fruit)   # 0 apple, 1 banana...\n\n"
            "person = {'name': 'Alice', 'age': 25}\n"
            "for key, value in person.items():\n"
            "    print(key, '=', value)\n\n"
            "squares = [x**2 for x in range(10)]\n"
            "evens = [x for x in range(20) if x % 2 == 0]"
        ),
        "tag": "control_flow"
    },
    {
        "patterns": [
            "while loop", "while in python", "infinite loop",
            "break continue", "loop control"
        ],
        "response": (
            "While Loops in Python:\n\n"
            "count = 0\n"
            "while count < 5:\n"
            "    print(count)\n"
            "    count += 1\n\n"
            "while True:\n"
            "    user = input('Enter password: ')\n"
            "    if user == 'secret':\n"
            "        break <- stop looping\n\n"
            "for i in range(10):\n"
            "    if i % 2 == 0:\n"
            "        continue <- skip even numbers\n"
            "    print(i) <- only prints odd numbers\n\n"
            "Avoid infinite loops!\n"
            "Always make sure the condition will eventually become False."
        ),
        "tag": "control_flow"
    },
    {
        "patterns": [
            "function", "how to create function", "def keyword",
            "define function", "return value", "function parameter",
            "function argument", "how to write function"
        ],
        "response": (
            "Functions in Python:\n\n"
            "def greet(name):\n"
            "    return f'Hello, {name}!'\n\n"
            "result = greet('Alice')\n"
            "print(result)          # Hello, Alice!\n\n"
            "def greet(name='World'):\n"
            "    return f'Hello, {name}!'\n\n"
            "print(greet())         # Hello, World!\n"
            "print(greet('Bob'))    # Hello, Bob!\n\n"
            "def min_max(numbers):\n"
            "    return min(numbers), max(numbers)\n\n"
            "low, high = min_max([3, 1, 4, 1, 5, 9])\n\n"
            "square = lambda x: x ** 2\n"
            "print(square(5))       # 25"
        ),
        "tag": "functions"
    },
    {
        "patterns": [
            "args kwargs", "*args", "**kwargs", "variable arguments",
            "multiple arguments", "unlimited arguments"
        ],
        "response": (
            "*args and **kwargs:\n\n"
            "def add_all(*args):\n"
            "    return sum(args)\n\n"
            "print(add_all(1, 2, 3, 4))\n"
            "print(add_all(10, 20))\n\n"
            "def describe(**kwargs):\n"
            "    for key, value in kwargs.items():\n"
            "        print(f'{key}: {value}')\n\n"
            "describe(name='Alice', age=25, city='Toronto')\n"
            "def everything(a, b, *args, **kwargs):\n"
            "    print(a, b, args, kwargs)"
        ),
        "tag": "functions"
    },
    {
        "patterns": [
            "list", "how to use list", "list methods", "append to list",
            "add to list", "remove from list", "list operations", "python list"
        ],
        "response": (
            "Lists in Python:\n\n"
            "fruits = ['apple', 'banana', 'cherry']\n\n"
            "print(fruits[0])\n"
            "print(fruits[-1])\n"
            "print(fruits[1:3])\n\n"
            "fruits.append('mango')\n"
            "fruits.insert(1, 'kiwi')\n"
            "fruits.remove('banana')\n"
            "fruits.pop()\n"
            "fruits.pop(0)\n\n"
            "len(fruits)\n"
            "fruits.sort()\n"
            "sorted(fruits)\n"
            "fruits.reverse()\n"
            "'apple' in fruits\n"
            "fruits.count('apple')\n"
            "fruits.index('cherry')"
        ),
        "tag": "data_structures"
    },
    {
        "patterns": [
            "dictionary", "dict", "how to use dictionary", "dict methods",
            "key value", "add to dict", "access dictionary", "python dict"
        ],
        "response": (
            "Dictionaries in Python:\n\n"
            "person = {\n"
            "    'name':  'Alice',\n"
            "    'age':   25,\n"
            "    'city':  'Toronto'\n"
            "}\n\n"
            "print(person['name'])           # Alice\n"
            "print(person.get('age'))        # 25\n"
            "print(person.get('email', 'N/A')) # N/A (safe default)\n\n"
            "person['email'] = 'a@b.com'    # add new key\n"
            "person['age'] = 26            # update value\n"
            "del person['city']              # delete key\n\n"
            "for key in person:\n"
            "    print(key)\n"
            "for key, val in person.items():\n"
            "    print(key, '->', val)\n\n"
            "person.keys()          # all keys\n"
            "person.values()        # all values\n"
            "'name' in person       # True/False"
        ),
        "tag": "data_structures"
    },
    {
        "patterns": [
            "try except", "error handling", "exception", "handle error",
            "try catch", "catch exception", "raise error", "finally"
        ],
        "response": (
            "Error Handling in Python:\n\n"
            "try:\n"
            "    number = int(input('Enter a number: '))\n"
            "    result = 10 / number\n"
            "    print(result)\n"
            "except ValueError:\n"
            "    print('That is not a valid number!')\n"
            "except ZeroDivisionError:\n"
            "    print('Cannot divide by zero!')\n"
            "except Exception as e:\n"
            "    print(f'Unexpected error: {e}')\n"
            "else:\n"
            "    print('Success!')  <- runs if NO exception\n"
            "finally:\n"
            "    print('Always runs') <- runs no matter what\n\n"
            "def set_age(age):\n"
            "    if age < 0:\n"
            "        raise ValueError('Age cannot be negative')\n"
            "    return age"
        ),
        "tag": "error_handling"
    },
    {
        "patterns": [
            "class", "object", "oop", "object oriented",
            "how to create class", "self keyword",
            "__init__", "constructor", "instance"
        ],
        "response": (
            "Classes and Objects in Python:\n\n"
            "class Dog:\n"
            "    def __init__(self, name, breed):\n"
            "        self.name  = name\n"
            "        self.breed = breed\n\n"
            "    def bark(self):\n"
            "        return f'{self.name} says: Woof!'\n\n"
            "    def describe(self):\n"
            "        return f'{self.name} is a {self.breed}'\n\n"
            "dog1 = Dog('Rex', 'Labrador')\n"
            "dog2 = Dog('Buddy', 'Poodle')\n\n"
            "print(dog1.bark())        # Rex says: Woof!\n"
            "print(dog2.describe())    # Buddy is a Poodle\n"
            "print(dog1.name)          # Rex\n\n"
            "class GuideDog(Dog):\n"
            "    def guide(self):\n"
            "        return f'{self.name} is guiding'"
        ),
        "tag": "oop"
    },
    {
        "patterns": [
            "inheritance", "parent class", "child class", "extend class",
            "super()", "override method", "polymorphism"
        ],
        "response": (
            "Inheritance in Python:\n\n"
            "class Animal:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n\n"
            "    def speak(self):\n"
            "        return 'Some sound'\n\n"
            "class Dog(Animal): <- inherits from Animal\n"
            "    def speak(self): <- overrides parent method\n"
            "        return 'Woof!'\n\n"
            "class Cat(Animal):\n"
            "    def speak(self):\n"
            "        return 'Meow!'\n\n"
            "animals = [Dog('Rex'), Cat('Whiskers')]\n"
            "for animal in animals:\n"
            "    print(animal.name, ':', animal.speak())\n"
            "class Puppy(Dog):\n"
            "    def __init__(self, name, toy):\n"
            "        super().__init__(name) <- call Dog.__init__\n"
            "        self.toy = toy"
        ),
        "tag": "oop"
    },
    {
        "patterns": [
            "read file", "write file", "open file", "file handling",
            "read csv", "with open", "file operations", "text file"
        ],
        "response": (
            "File Handling in Python:\n\n"
            "with open('data.txt', 'r') as f:\n"
            "    content = f.read()  <- entire file\n"
            "    lines = f.readlines() <- list of lines\n\n"
            "with open('output.txt', 'w') as f:\n"
            "    f.write('Hello World\\n')\n\n"
            "with open('log.txt', 'a') as f:\n"
            "    f.write('New line\\n')\n\n"
            "with open('data.txt', 'r') as f:\n"
            "    for line in f:\n"
            "        print(line.strip())\n\n"
            "import pandas as pd\n"
            "df = pd.read_csv('data.csv')\n"
            "print(df.head())\n\n"
            "✔ Always use 'with open' — it closes the file automatically!"
        ),
        "tag": "files"
    },
    {
        "patterns": [
            "indentation error", "indentationerror",
            "invalid syntax", "syntaxerror", "common error",
            "syntax error", "indent"
        ],
        "response": (
            "Common Python Errors:\n\n"
            "IndentationError:\n"
            " - Python uses indentation (spaces) to define blocks\n"
            " - Always use 4 spaces (not tabs) consistently\n"
            "✘ if True:\n"
            "✘ print('hello') <- must be indented!\n"
            "✔ if True:\n"
            "✔     print('hello')\n\n"
            "SyntaxError:\n"
            " - Missing colon after if/for/def/class\n"
            " - Missing closing bracket or quote\n"
            "✘ if x > 5\n"
            "✔ if x > 5:\n\n"
            "NameError:\n"
            " - Using a variable before defining it\n"
            "✘ print(name) <- name not defined yet!\n"
            "✔ name = 'Alice'\n"
            "✔ print(name)\n\n"
            "TypeError:\n"
            " - Wrong type in operation\n"
            "✘ print('Age: ' + 25) <- can't add str + int\n"
            "✔ print('Age: ' + str(25))\n"
            "✔ print(f'Age: {25}')"
        ),
        "tag": "errors"
    },
    {
        "patterns": [
            "indexerror", "index error", "list index out of range",
            "keyerror", "key error", "attributeerror", "typeerror"
        ],
        "response": (
            "More Common Errors:\n\n"
            "IndexError — list index out of range:\n"
            "fruits = ['apple', 'banana'] <- only index 0 and 1\n"
            "✘ print(fruits[5]) <- index 5 doesn't exist!\n"
            "✔ print(fruits[-1]) <- last item safely\n"
            "✔ if len(fruits) > 5: print(fruits[5])\n\n"
            "KeyError — dictionary key not found:\n"
            "person = {'name': 'Alice'}\n"
            "✘ print(person['age']) <- 'age' key doesn't exist!\n"
            "✔ print(person.get('age', 'Unknown'))\n\n"
            "AttributeError — object has no such method:\n"
            "✘ name = 'hello'\n"
            "✘ name.append('world') <- strings have no append!\n"
            "✔ use list.append() for lists\n\n"
            "TypeError — wrong type:\n"
            "✘ '5' + 5    <- can't add str and int\n"
            "✔ int('5') + 5  or  '5' + str(5)"
        ),
        "tag": "errors"
    },
    {
        "patterns": [
            "import", "how to install package", "pip install",
            "library", "module", "import numpy", "import pandas",
            "how to import"
        ],
        "response": (
            "pip install numpy\n"
            "pip install pandas scikit-learn matplotlib\n\n"
            "import numpy\n"
            "numpy.array([1, 2, 3])\n\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "np.array([1, 2, 3])\n\n"
            "from math import sqrt, pi\n"
            "print(sqrt(16))\n"
            "print(pi)\n\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from sklearn.linear_model import LinearRegression"
        ),
        "tag": "libraries"
    },
    {
        "patterns": [
            "hello", "hi", "hey", "good morning",
            "good evening", "howdy", "sup", "what's up"
        ],
        "response": (
            "Hey there! I'm your Python Programming Assistant!\n\n"
            "I can help you with:\n"
            "Python basics\n"
            "Control flow\n"
            "Functions\n"
            "Data structures\n"
            "Error handling\n"
            "OOP\n"
            "File handling\n"
            "Debugging common errors\n"
            "Libraries and imports\n\n"
            "What do you need help with today?"
        ),
        "tag": "greeting"
    },
    {
        "patterns": [
            "help", "what can you do", "topics", "what do you know",
            "menu", "options", "commands"
        ],
        "response": (
            "Topics I can help with:\n\n"
            "Say things like:\n"
            " - 'How do I print in Python?'\n"
            " - 'What is a for loop?'\n"
            " - 'How do I create a function?'\n"
            " - 'What is a dictionary?'\n"
            " - 'How do I handle errors?'\n"
            " - 'What is a class?'\n"
            " - 'How do I read a file?'\n"
            " - 'What are common Python errors?'\n"
            " - 'How do I import a library?'\n\n"
            "Just ask naturally — I'll understand!"
        ),
        "tag": "meta"
    },
    {
        "patterns": [
            "thank you", "thanks", "thank you so much",
            "that helped", "great", "awesome", "perfect"
        ],
        "response": (
            "You're welcome! Happy coding!\n\n"
            "Remember:\n"
            " - Practice every day — even 20 minutes helps\n"
            " - Read error messages carefully — they tell you exactly what's wrong\n"
            " - Google is your best friend as a developer\n"
            " - Break big problems into small ones\n\n"
            "Ask me anything else anytime!"
        ),
        "tag": "meta"
    },
    {
        "patterns": [
            "bye", "goodbye", "quit", "exit", "see you",
            "see ya", "later", "good night"
        ],
        "response": (
            "Goodbye! Keep coding and keep learning!\n\n"
            "'The best way to learn to code is to code.'\n\n"
            "Come back anytime you have Python questions!"
        ),
        "tag": "farewell"
    },
]

# Step 2 — rule-based chatbot
class RuleBasedChatbot:
    """
    Rule-Based Chatbot — using pattern matching.

    how it works: -> Checks if any known pattern appears in the user's message -> If found, returns the matching response -> Fast and predictable but only matches exact patterns
    """
    def __init__(self, knowledges):
        self.kb = knowledges

    def clean(self, text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)

        return text

    def respond(self, user_input):
        cleaned = self.clean(user_input)

        for item in self.kb:
            for pattern in item['patterns']:
                # check if pattern words appear in user input
                pattern_words = pattern.lower().split()
                if all(word in cleaned for word in pattern_words):
                    return item['response']

                # check if user input appears in pattern
                if cleaned in pattern.lower():
                    return item['response']

        return None # no match found

# Step 3 — ML-based chatbot
class MLChatbot:
    """
    ML-Based Chatbot — uses TF-IDF and cosine similarity.

    How it works: -> 
        Converts all known patterns into TF-IDF vectors -> 
        Converts user input into a TF-IDF vector -> 
        Finds the pattern most similar to the user input -> 
        Returns the matching response -> 
        Handles paraphrasing and variations better than rule-based
    """

    def __init__(self, knowledges, threshold=0.15):
        self.kb    = knowledges
        self.threshold = threshold
        self.patterns  = []
        self.responses = []
        self._build_index()

    def _build_index(self):
        """Build TF-IDF index from all patterns."""

        for item in self.kb:
            for pattern in item['patterns']:
                self.patterns.append(pattern)
                self.responses.append(item['response'])

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.pattern_vectors = self.vectorizer.fit_transform(self.patterns)

    def respond(self, user_input):
        # vectorise user input
        try:
            user_vector = self.vectorizer.transform([user_input.lower()])
        except Exception:
            return None

        # calculate similarity with all patterns
        similarities = cosine_similarity(user_vector, self.pattern_vectors)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score >= self.threshold:
            return self.responses[best_idx], best_score
        return None, best_score

# Step 4 — combined chatbot
class PythonHelpChatbot:
    """
    Combined chatbot — uses rule-based first, ML as fallback.

    Strategy: -> 
        Try rule-based first (fast, precise) -> 
        If no match, try ML (handles paraphrasing) -> 
        If still no match, return a helpful fallback message
    """

    def __init__(self, knowledges):
        print("Initialising rule-based engine...")
        self.rule_bot = RuleBasedChatbot(knowledges)

        print("Building ML index (TF-IDF)...")
        self.ml_bot = MLChatbot(knowledges, threshold=0.15)

        self.conversation_history = []
        print("✔ Chatbot ready!\n")

    def respond(self, user_input, mode='combined'):
        """
        mode: 
            'rule' -> only rule-based 
            'ml' -> only ML-based 
            'combined-> rule first, ML fallback
        """
        user_input = user_input.strip()
        if not user_input:
            return "Please type a question!", "none"

        # check for quit
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            return self.rule_bot.respond(user_input) or "Goodbye! 👋", "rule"

        if mode == 'rule':
            response = self.rule_bot.respond(user_input)
            if response:
                return response, "rule"
            return self._fallback(user_input), "fallback"

        elif mode == 'ml':
            response, score = self.ml_bot.respond(user_input)
            if response:
                return response, f"ml ({score:.2f})"
            return self._fallback(user_input), "fallback"

        else: # combined
            # try rule-based first
            response = self.rule_bot.respond(user_input)
            if response:
                return response, "rule"

            # fall back to ml
            response, score = self.ml_bot.respond(user_input)
            if response:
                return response, f"ml ({score:.2f})"

            return self._fallback(user_input), "fallback"

    def _fallback(self, user_input):
        fallbacks = [
            (
                "I'm not sure I understand that one.\n\n"
                "Try asking about:\n"
                " - Python basics, variables, data types\n"
                " - For loops, while loops, if/else\n"
                " - Functions, classes, file handling\n"
                " - Common errors and how to fix them\n"
                " - Libraries and imports\n\n"
                "Or type 'help' to see all topics!"
            ),
            (
                "Hmm, I don't have an answer for that yet.\n\n"
                "I'm specialised in Python programming help.\n"
                "Try rephrasing or ask about a specific Python topic!"
            ),
            (
                "I didn't quite catch that.\n\n"
                "Some example questions I can answer:\n"
                "'How do I create a list?'\n"
                "'What is a class in Python?'\n"
                "'How do I handle errors?'\n"
                "'What is the difference between a list and a tuple?'"
            )
        ]
        return random.choice(fallbacks)

# Step 5 — Demo & Comparison
def run_demo(chatbot):
    print("Rule-based vs ML-based responses")

    test_questions = [
        "How do I print in Python?",
        "show me a loop example",
        "I keep getting IndentationError",
        "what is self in a class",
        "how to make a function",
        "dictionary vs list",
        "help me understand try except",
    ]

    print(f"\n{'Question':<40} {'Rule-Based':>12} {'ML-Based':>12}\n\n")

    for question in test_questions:
        rule_resp = chatbot.rule_bot.respond(question)
        ml_resp, ml_score = chatbot.ml_bot.respond(question)

        rule_result = "✔ Match" if rule_resp else "✘ No match"
        ml_result = f"✔ {ml_score:.2f}" if ml_resp else f"✘ {ml_score:.2f}"

        short_q = question[:38] + ".." if len(question) > 38 else question

        print(f"{short_q:<40} {rule_result:>12} {ml_result:>12}")

    print("\nKey observations:")
    print(" - Rule-based: fast but only matches exact patterns")
    print(" - ML-based: handles paraphrasing and variations better")
    print(" - Combined: best of both worlds!")

# Step 6 — Interactive chat
def run_chat(chatbot):
    print("Powered by Rule-Based + ML (TF-IDF Similarity)")
    print("\n\nAsk me anything about Python programming!")
    print("Type 'demo' to see a comparison, 'quit' to exit.\n\n")

    while True:
        try:
            user_input = input("Enter your question: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'demo':
                run_demo(chatbot)

                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                response, _ = chatbot.respond(user_input)
                print(f"\nBot: {response}")

                break

            response, source = chatbot.respond(user_input)

            print(f"\nBot [{source}]:\n{response}")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nBot: Goodbye!")

            break

if __name__ == '__main__':
    print("Initialising chatbot...\n")

    bot = PythonHelpChatbot(knowledges)

    run_demo(bot)
    run_chat(bot)
