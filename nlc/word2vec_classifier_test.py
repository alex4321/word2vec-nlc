import unittest
import os
from .classes import Classes
from .word2vec_classifier import Word2VecClassifier


class Word2VecClassifierTest(unittest.TestCase):
    def _classifier_data(self):
        stop_words = [
            "a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually",
            "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost",
            "alone", "along", "already", "also", "although", "always", "am", "among", "amongst",
            "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway",
            "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't",
            "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away",
            "awfully", "b", "be", "became", "because", "become", "becomes", "becoming", "been",
            "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides",
            "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon",
            "c's", "came", "cause", "causes", "certain", "certainly", "changes", "clearly", "co",
            "com", "come", "comes", "concerning", "consequently", "consider", "considering",
            "contain", "containing", "contains", "corresponding", "could", "couldn't", "course",
            "currently", "d", "definitely", "described", "despite", "did", "didn't", "different",
            "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else",
            "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every",
            "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example",
            "except", "f", "far", "few", "fifth", "first", "five", "followed", "following",
            "follows", "for", "former", "formerly", "forth", "four", "from", "further",
            "furthermore", "g", "get", "gets", "getting", "given", "gives", "go", "goes", "going",
            "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly", "has",
            "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her",
            "here", "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
            "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i",
            "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored", "immediate", "in", "inasmuch",
            "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead",
            "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "j",
            "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately",
            "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked",
            "likely", "little", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may",
            "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most",
            "mostly", "much", "must", "my", "myself", "n", "name", "namely", "nd", "near",
            "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new",
            "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not",
            "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh",
            "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other",
            "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "over",
            "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed",
            "please", "plus", "possible", "presumably", "probably", "provides", "q", "que", "quite",
            "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless",
            "regards", "relatively", "respectively", "right", "s", "said", "same", "saw",
            "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemedould",
            "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone",
            "something", "sometime", "sometimes", "seeming", "seems", "seen", "self", "selves",
            "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "sh",
            "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still",
            "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends", "th",
            "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their",
            "theirs", "them", "themselves", "then", "thence", "there", "there's", "thereafter",
            "thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "they'd",
            "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly",
            "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together",
            "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice",
            "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up",
            "upon", "us", "use", "used", "useful", "uses", "using", "usually", "uucp", "v", "value",
            "various", "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way",
            "we", "we'd", "we'll", "we're", "we've", "welcome", "well", "went", "were", "weren't",
            "whatever", "when", "whence", "whenever", "where", "where's", "whereafter", "whereas",
            "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
            "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish",
            "with", "within", "without", "won't", "wonder", "would", "wouldn't", "x", "y", "yes",
            "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
            "yourselves", "z", "zero"
        ]
        path = os.path.join(os.path.dirname(__file__), "test", "glove.6B.50d.bin")
        classes = Classes({
            "capabilities": [
                "can you turn on the radio", "help", "help me",
                "How are you going to help me?", "how can you help me",
                "need help", "tell me what you can do", "What are you",
                "what are you capable of", "what are your capabilities Watson?",
                "what can I do", "What can i say", "what can you do",
                "what can you do for me", "what can you help me with",
                "what do you know", "what else", "what things can you do"
            ],
            "locate_amenity": [
                "Amenities", "can you find a good place to eat?",
                "can you find the nearest gas station for me",
                "find a restaurant", "Find best gas option", "find me a restaurant",
                "get me directions", "I am hungry", "I'd like to get something to eat",
                "i'm hungry", "im hungry I want to eat something", "i need some coffee",
                "i need to stop for gas", "Information on locations near me", "I want a bar",
                "i want drink", "I want eat", "i want pizza", "I want to eat pizza",
                "I want to stop for some food", "let's get some grub", "locate ammenity",
                "Navigation", "nearby restaurant", "order a pizza", "pizza", "restaurant",
                "restaurants", "show me the best path", "Stop for food/coffee",
                "where are the closest restrooms", "where can i drink", "where can i eat pizza?",
                "where can i have pizza", "where is a restaurant", "where is ATM",
                "where is the bathroom?", "where is the pool", "where's the nearest exit"
            ],
            "out_of_scope": [
                "Can I play music through bluetooth from my phone?", "change bulb",
                "changing gears", "changing the light bulb", "checking the engine oil",
                "coolant level", "fuel tank flap", "hi to switch on the fog lights",
                "how can i change oil", "how can i connect the phone",
                "how can i find current level of oil", "how can i lock the car",
                "how can i replace light bulb", "how do i adjust exterior mirrors",
                "how do i adjust head restraint", "how do i adjust the cruise control speed",
                "how do i adjust the mirror", "how do i adjust the mirror inside",
                "how do i adjust the rear view mirror", "how do i adjust the vehicle distance",
                "how do i arm the alarm", "how do i browse the tracks on the iPod",
                "how do i cancel cruise control", "how do i change oil",
                "how do i change the headlight bulbs", "how do i change the lights bulb",
                "how do i change the oil", "how do i clean the vehicle",
                "how do i close the window", "how do i connect my phone",
                "how do i connect USB device", "how do i control a phone",
                "how do i disable the vehicle stability control",
                "how do i fill in the engine oil", "how do i find the fuel consumption",
                "how do i lock the window", "how do i lock the windows",
                "how do i lock the windows to prevent children from opening",
                "how do i open the doors", "how do i open the fuel filler flap",
                "how do i open the power windows", "how do i open the rear window",
                "how do i open the side window", "how do i open the tank fuel"
            ]
        })
        return stop_words, classes, path

    def test_classifier(self):
        stop_words, classes, path = self._classifier_data()
        classifier = Word2VecClassifier(path, stop_words)
        classifier.train(classes)
        confidences = classifier.classify("So, what I can do?")
        self.assertGreater(confidences['capabilities'],
                           max(confidences['locate_amenity'], confidences['out_of_scope']))
        confidences = classifier.classify("Find place")
        self.assertGreater(confidences['locate_amenity'],
                           max(confidences['capabilities'], confidences['out_of_scope']))

    def test_config(self):
        stop_words, classes, path = self._classifier_data()
        classifier = Word2VecClassifier(path, stop_words)
        classifier.train(classes)
        config = classifier.config()
        classifier = Word2VecClassifier(config=config)
        confidences = classifier.classify("So, what I can do?")
        self.assertGreater(confidences['capabilities'],
                           max(confidences['locate_amenity'], confidences['out_of_scope']))
        confidences = classifier.classify("Find place")
        self.assertGreater(confidences['locate_amenity'],
                           max(confidences['capabilities'], confidences['out_of_scope']))
