from sentimental_hwglu.words_statistics import WordStatistics
from sentimental_hwglu.utils import tokenizer_porter
from sentimental_hwglu.words_statistics import WordStatistics
from sklearn.feature_extraction.text import CountVectorizer

# clean words from: https://parade.com/1241177/marynliles/positive-words/
_positive_words = [
    "amaizing", "great", "greatest", "entertained", "entertain",
    "love",
    "absolutely", "abundant", "accessible", "acclaimed", "accommodative",
    "achievement", "adaptive", "admire", "adore", "adulation", "affability",
    "agathist", "alive", "amuse", "animated", "approve", "assure",
    "attractive", "awesome", "positive", "baronial", "beaming", "beautiful",
    "beguiling", "beloved", "benignant", "best", "bewitching", "boss",
    "brainy", "breathtaking", "bubbly", "related:", "positive", "centered",
    "champion", "charismatic", "charming", "cheerful", "chic", "chipper",
    "chummy", "classy", "clever", "colorful", "comical", "communicative",
    "constant", "courageous", "positive", "definite", "delectable", "delicious",
    "delightful", "dependable", "dignified", "divine", "down-to-earth",
    "dreamy", "dynamite", "positive", "ecstatic", "electrifying", "employable",
    "empowered", "endearing", "enjoyable", "enriching", "enthusiastic",
    "enticing", "especial", "excellent", "exciting", "exhilarating", "exultant",
    "positive", "fab", "fain", "fantastic", "fashionable", "favorite",
    "fearless", "fetching", "fiery", "friend", "fun", "positive", "gallant",
    "gay", "genuine", "gifted", "gleaming", "glittering", "gnarly", "goodhearted",
    "grandiose", "greatest", "gumptious", "positive", "happy", "heavenly",
    "honorable", "hospitable", "humanitarian", "hypnotic", "positive", "ideal",
    "imaginative", "impeccable", "impressive", "incredible", "innovative",
    "insightful", "inspiring", "instinctive", "intellectual", "irresistible",
    "positive", "Jammy", "Jesting", "Jolly", "Jovial", "Joysome", "Judicious",
    "Juicy", "Just", "positive", "Keen", "Kind-hearted", "Knightly", "Knockout",
    "Knowledgeable", "positive", "laid-back", "lambent", "laudable", "legendary",
    "level-headed", "likable", "lionhearted", "lively", "lovely", "luminous",
    "positive", "magical", "magnetic", "magnificent", "majestic", "marvelous",
    "masterful", "mindful", "miraculous", "motivated", "moving", "positive",
    "neighborly", "nifty", "noble", "numinous", "positive", "obedient", "obliging",
    "observant", "on-target", "open-hearted", "open-minded", "optimistic",
    "orderly", "organized", "original", "outgoing", "out-of-this-world",
    "outstanding", "overjoyed", "positive", "pally", "paramount", "passionate",
    "patient", "peaceful", "peachy", "peppy", "perceptive", "persevering", "persistent",
    "personable", "persuasive", "phenomenal", "philanthropic", "picturesque",
    "piquant", "playful", "polished", "posh", "prized", "proactive", "promising",
    "proud", "punctual", "positive", "queenly", "quick-witted", "quirky", "positive",
    "rad", "radiant", "rapturous", "razor-sharp", "reassuring", "recherche",
    "recommendable", "refulgent", "reliable", "remarkable", "resilient", "resourceful",
    "respectable", "revolutionary", "positive", "saccharine", "sagacious", "savvy",
    "self-assured", "sensational", "sincere", "snappy", "snazzy", "spellbinding",
    "splendiferous", "spunky", "stellar", "striking", "positive", "teeming",
    "tender-hearted", "thoughtful", "thriving", "timeless", "tolerant", "trailblazing",
    "transcendental", "tubular", "positive", "upbeat", "uplifting", "upstanding",
    "urbane", "positive", "valiant", "vibrant", "victorious", "visionary",
    "vivacious", "positive", "warm", "well-read", "whimsical", "whiz-bang", "wholehearted",
    "winsome", "wise", "witty", "wizardly", "wondrous", "worldly", "positive",
    "xenial", "xenodochial", "positive", "Yay", "Yes", "Yummiest", "positive", "zappy",
    "zazzy", "zealful", "zealous", ]

_negative_words = [
    "dumb", "dumber", "dumbest"
    "abrasive", "apathetic", "controlling", "dishonest", "impatient", "anxious", "betrayed",
    "disappointed", "embarrassed", "Jealous", "abysmal", "bad", "callous", "corrosive",
    "damage", "despicable", "don’t", "enraged", "fail", "gawky", "haggard", "hurt", "icky",
    "insane", "Jealous", "lose", "malicious", "naive", "not", "objectionable", "pain",
    "questionable", "reject", "rude", "sad", "sinister", "stuck", "tense", "ugly", "unsightly",
    "vice", "wary", "Yell", "zero", "adverse", "banal", "can’t", "corrupt", "damaging", "detrimental",
    "dreadful", "eroding", "faulty", "ghastly", "hard", "hurtful", "ignorant", "insidious", "Junky",
    "lousy", "mean", "nasty", "noxious", "odious", "perturb", "quirky", "renege", "ruthless",
    "savage", "slimy", "stupid", "terrible", "undermine", "untoward", "vicious", "weary", "Yucky",
    "alarming", "barbed", "clumsy", "dastardly", "dirty", "dreary", "evil", "fear", "grave", "hard-hearted",
    "ignore", "injure", "insipid", "lumpy", "menacing", "naughty", "none", "offensive", "pessimistic",
    "quit", "repellant", "scare", "smelly", "substandard", "terrifying", "unfair", "unwanted", "vile",
    "wicked", "angry", "belligerent", "coarse", "crazy", "dead", "disease", "feeble", "greed",
    "harmful", "ill", "injurious", "messy", "negate", "no", "old", "petty", "reptilian", "scary", "sobbing",
    "suspect", "threatening", "unfavorable", "unwelcome", "villainous", "woeful", "annoy", "bemoan", "cold",
    "creepy", "decaying", "disgusting", "fight", "grim", "hate", "immature", "misshapen", "negative",
    "nothing", "oppressive", "plain", "repugnant", "scream", "sorry", "suspicious", "unhappy",
    "unwholesome", "vindictive", "worthless", "anxious", "beneath", "cold-hearted", "criminal", "deformed",
    "disheveled", "filthy", "grimace", "hideous", "imperfect", "missing", "never", "neither", "poisonous",
    "repulsive", "severe", "spiteful", "unhealthy", "unwieldy", "wound", "apathy", "boring", "collapse", "cruel",
    "deny", "dishonest", "foul", "gross", "homely", "impossible", "misunderstood", "no", "nowhere",
    "poor", "revenge", "shocking", "sticky", "unjust", "unwise", "appalling", "broken", "confused", "cry",
    "deplorable", "dishonorable", "frighten", "grotesque", "horrendous", "inane", "moan", "nobody", "prejudice",
    "revolting", "shoddy", "stinky", "unlucky", "upset", "atrocious", "contrary", "cutting", "depressed",
    "dismal", "frightful", "gruesome", "horrible", "inelegant", "moldy", "nondescript", "rocky", "sick",
    "stormy", "unpleasant", "awful", "contradictory", "deprived", "distress", "guilty", "hostile", "infernal",
    "monstrous", "nonsense", "rotten", "sickening", "stressful", "unsatisfactory",
]

negations = [
    "not", 
    "isn't", "aren't", "doesn't", "didn't", "couldn't", "shouldn't", "won't", "ain't"
    "isn",   "aren",   "doesn",   "didn",   "couldn",   "shouldn",            "ain"
]

def is_good(word: str, positive_words) -> bool:
    return word.lower() in positive_words

def is_bad(word: str, negative_words) -> bool:
    return word.lower() in negative_words

class StatDiff:
    def __init__(self, word, use0, use1) -> None:
        self.word = word
        self.use0 = use0
        self.use1 = use1

class NaiveSA:
    """
    Naive class to performe Sentimental Analysis: 
    
    it counts the number of good words (n_g) and bad words (n_b).
    If the count of good words is higher then the bad words, then the sentence is predict to be positive,
    otherwise is negative.

    A negation can influence up to 3 following words. 
    In the sentence: it is not very good,
    the good word 'good' is interpreted as bad.
    similarly, in the sentence, "it is not very ugly"
    the bad word 'ugly' is interpreted as good.
    """
    _max_not_counter = 3
    def __init__(self, verbose=False, tokenizer=None) -> None:
        self._verbose = verbose
        self._sets = []
        self._vocabularies = []
        self._aggregations = []
        self._diff_use_stats = {}
        self._set0 = set()
        self._set1 = set()
        self._set_common = set()
        self._list_positive_words = {}
        self._list_negtive_words = {}
        self._tokenizer = tokenizer if tokenizer is not None else tokenizer_porter
        self._base_positive_words = self._tokenizer(' '.join(_positive_words))
        self._base_negative_words = self._tokenizer(' '.join(_negative_words))
        self._negations = self._tokenizer(' '.join(negations))

    def _predict_sentence(self, words) -> int:
        not_counter = 0
        n_good, n_bad = 0, 0
        for word in self._tokenizer(words):
            if word in self._list_negtive_words:
                n_bad += self._list_negtive_words[word] / 2
            elif word in self._list_positive_words:
                n_good += self._list_positive_words[word] / 2
            if len(word) < 2: continue
            if word == "and": continue
            not_counter -= 1
            if word in self._negations:
                not_counter = self._max_not_counter
                continue
            word_is_good = is_good(word, self._base_positive_words)
            word_is_bad = is_bad(word, self._base_negative_words) if not word_is_good else False
            if (word_is_good and not_counter <= 0) or (word_is_bad and not_counter > 0):
                n_good += 1
                not_counter = 0
            if (word_is_bad and not_counter <= 0) or (word_is_good and not_counter > 0):
                n_bad += 1
                not_counter = 0
        # if self._verbose: print("n_good: {}, n_bad: {}".format(n_good, n_bad))
        return 1 if n_good > n_bad else 0

    def fit(self, X, y):
        if self._verbose: print(" fitting data - size: ", len(y))
        self._create_sets_of_words(X, y)
        # self._create_freq_maps_list()
        self._create_diff_use_stats_of_words()
        self._create_positive_and_negative_wordset()
    
    def _create_sets_of_words(self, X, y):
        if self._verbose: print(" create sets of words")
        for k, filter in enumerate([lambda x : x == 0, lambda x : x == 1]):
            if self._verbose: print("    - Processing ", k)
            sw = WordStatistics()
            # sw.words2Vect(X, vectorizer=CountVectorizer, tokenizer=tokenizer_porter)
            sw.words2VectFiltered(X, y, vectorizer=CountVectorizer, tokenizer=self._tokenizer, filter=filter)
            self._sets.append(set(sw.top(n=-1)))
            self._vocabularies.append(sw._vect.vocabulary_)
            self._aggregations.append(sw.getAggregated())

        self._set0 = self._sets[0].difference(self._sets[1])
        self._set1 = self._sets[1].difference(self._sets[0])
        self._set_common = self._sets[0].intersection(self._sets[1])
    
    def _create_positive_and_negative_wordset(self):
        if self._verbose: print(" create_positive_and_negative_wordse")
        a = list(self._diff_use_stats.keys())
        a.sort()
        self._list_positive_words = {}
        self._list_negtive_words = {}
        positive = lambda z: z.use1
        negative = lambda z: z.use0
        for name, func in [["NEGATIVE", negative], ["POSITIVE", positive]]:
            for k in a:
                stats = self._diff_use_stats[k]
                for u in stats:
                    tot = u.use1 + u.use0
                    value = func(u)
                    perc = value / (tot)
                    if perc >= 0.80 and value > 100:
                        print("[{}]".format(name), u.word, ": ", func(u), ' -> {:.3f}%'.format(perc))
                        if name == "NEGATIVE": 
                            self._list_negtive_words[u.word] = perc 
                        else: 
                            self._list_positive_words[u.word] = perc
        if self._verbose: print("     {} new positive words".format(len(self._list_positive_words)))
        if self._verbose: print("     {} new negative words".format(len(self._list_negtive_words)))

    def _create_freq_maps_list(self):
        if self._verbose: print(" create_freq_maps_list")
        freq_maps_list = []
        freq_map = {}
        for k in self._set0: 
            aggregates = self._aggregations[0][self._vocabularies[0][k]]
            freq_map[k] = aggregates
        freq_maps_list.append(freq_map)

        freq_map = {}
        for k in self._set1: 
            aggregates = self._aggregations[1][self._vocabularies[1][k]]
            freq_map[k] = aggregates
        freq_maps_list.append(freq_map)
    
    def _create_diff_use_stats_of_words(self):
        if self._verbose: print(" create_diff_use_stats_of_words")
        self._diff_use_stats = {}
        for k in self._set_common: 
            use_1 = self._aggregations[1][self._vocabularies[1][k]]
            use_0 = self._aggregations[0][self._vocabularies[0][k]]
            try:
                self._diff_use_stats[use_1 - use_0].append(StatDiff(k, use_0, use_1))
            except:
                self._diff_use_stats[use_1 - use_0] = [StatDiff(k, use_0, use_1)]

    def score(self, X, y):
        if self._verbose: print(" calculate score of the model")
        n, n_correct = 0, 0
        tot = len(y)
        done = 0
        for x, z in zip(X, y):
            done += 1
            if self._verbose: print(" processing {} / {} ({} %)".format(done, tot, done * 100 // tot, 100), end='\r')
            n += 1
            if self._predict_sentence(x) == z:
                n_correct += 1
        return (float)(n_correct) / (float)(n) if n > 0 else 0
    
    def predict(self, X):
        return self._predict_sentence(X)
