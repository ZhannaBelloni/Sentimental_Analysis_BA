import json
from sentimental_hwglu.utils import tokenizer_porter
from sentimental_hwglu.words_statistics import WordStatistics, WordStatisticsVect
from sklearn.feature_extraction.text import CountVectorizer
import jdata as jd;
from sklearn.pipeline import Pipeline

import numpy as np

from sentimental_hwglu.sa_model import SentimentAnalysisPipeline

# note: 
# https://parade.com/1241177/marynliles/positive-words/
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
    "damage", "despicable", "don't", "enraged", "fail", "gawky", "haggard", "hurt", "icky",
    "insane", "Jealous", "lose", "malicious", "naive", "not", "objectionable", "pain",
    "questionable", "reject", "rude", "sad", "sinister", "stuck", "tense", "ugly", "unsightly",
    "vice", "wary", "Yell", "zero", "adverse", "banal", "can't", "corrupt", "damaging", "detrimental",
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

def tokeniezer_default(x) : 
    return [z.strip() for z in x.split() if len(z.strip()) > 0]
class StatDiff:
    def __init__(self, word, use0, use1) -> None:
        self.word = word
        self.use0 = use0
        self.use1 = use1

class NaiveSA(SentimentAnalysisPipeline):
    _max_not_counter = 3
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.set_params(**kwargs)
        self._sets = []
        self._vocabularies = []
        self._aggregations = []
        self._diff_use_stats = {}
        self._set0 = set()
        self._set1 = set()
        self._set_common = set()
        self._list_positive_words = {}
        self._list_negative_words = {}
        self._base_positive_words = self._tokenizer(' '.join(_positive_words))
        self._base_negative_words = self._tokenizer(' '.join(_negative_words))
        self._negations = self._tokenizer(' '.join(negations))
        self._pipeline = self


    def _predict_sentence(self, words) -> int:
        not_counter = 0
        n_good, n_bad = 0, 0
        for word in self._tokenizer(words):
            if word in self._list_negative_words:
                n_bad += self._list_negative_words[word] * self._weigth_added_words
            elif word in self._list_positive_words:
                n_good += self._list_positive_words[word] * self._weigth_added_words
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
        # if self._verbose > 0: print("n_good: {}, n_bad: {}".format(n_good, n_bad))
        return 1 if n_good > n_bad else 0
    
    def set_params(self, **kwargs):
        self._params = kwargs
        self._weigth_added_words = float(kwargs.get('weigth_added_words', 1.0))
        self._verbose = kwargs.get('verbose', 0)
        self._tokenizer_name = kwargs.get("tokenizer_name", 'split')
        if self._tokenizer_name == 'tokenizer_porter': self._tokenizer = tokenizer_porter 
        else: self._tokenizer = tokeniezer_default
        self._use_frequency = kwargs.get('use_frequency', False)
        return self

    def get_params(self, deep=None): 
        params = self._params
        params['weigth_added_words'] = self._weigth_added_words
        params['verbose'] = self._verbose
        params['tokenizer_name'] = self._tokenizer_name
        params['use_frequency'] = self._use_frequency
        return params

    def fit(self, X, y, **fitparams):
        if self._verbose > 0: print(" fit: params -> ", self.get_params())
        if self._weigth_added_words == 0: 
            print(" weigth_added_words set to 0: no fitting"); 
            return
        if self._verbose > 0: print(" fitting data - size: ", len(y))
        self._create_sets_of_words(X, y)
    
    def _create_sets_of_words(self, X, y):
        ws = WordStatistics(df=None, X=X, y=y)
        ws.createWordsSets()
        df = ws.commonWordsDataFrame()
        pos = df.sort_values('count_use', ascending=False).sort_values('fraction_positive', ascending=False).query("fraction_positive > 0.7")['word'].values
        neg = df.sort_values('count_use', ascending=False).sort_values('fraction_negative', ascending=False).query("fraction_negative > 0.7")['word'].values
        self._list_positive_words = {}
        self._list_negative_words = {}
        for p in self._tokenizer(' '.join(pos)):
            if not self._use_frequency: self._list_positive_words[p] = 1 
            else:
                try: self._list_positive_words[p] = df.query("word == '{}'".format(p))['fraction_positive'].values[0]
                except: pass
        for n in self._tokenizer(' '.join(neg)): 
            if not self._use_frequency: self._list_negative_words[n] = 1 
            else:
                try: self._list_negative_words[n] = df.query("word == '{}'".format(n))['fraction_positive'].values[0]
                except: pass

    
    def score(self, X, y):
        if self._verbose > 1: print(" calculate score of the model")
        n, n_correct = 0, 0
        tot = len(y)
        done = 0
        for x, z in zip(X, y):
            done += 1
            n += 1
            if self._verbose > 1: print(" processing {} / {} ({} %)".format(done, tot, done * 100 // tot, 100), end='\r')
            if self._predict_sentence(x) == z:
                n_correct += 1
        if self._verbose > 1: print("")
        s = (float)(n_correct) / (float)(n) if n > 0 else 0
        if self._verbose > 1: print(" score: ", s, "\n")
        return s

    def predict(self, X):
        if self._verbose > 0: print(" predict model on {} data".format(len(X)))
        if self._verbose > 1: print(" predict: params -> ", self.get_params())
        n, n_correct = 0, 0
        tot = len(X)
        done = 0
        y_pred = []
        for x in X:
            n += 1
            done += 1
            if self._verbose > 1: print(" processing {} / {} ({} %)".format(done, tot, done * 100 // tot, 100), end='\r')
            y_pred.append(self._predict_sentence(x))
        if self._verbose > 0: print("")
        return np.array(y_pred)

    def predict_proba(self, data):
        return self.predict(data)

    def transorm(self, data):
        return data

    def fit_transorm(self, data, target):
        X = self.transorm(data)
        self.fit(X, target)

    def save(self, fileout):
        if fileout is None: return
        model = {
            "weigth_added_words": self._weigth_added_words,
            "verbose": self._verbose,
            "tokenizer": str(self._tokenizer),
            "use_frequency": self._use_frequency ,
            "list_positive_words": self._list_positive_words,
            "list_negative_words": self._list_negative_words,
            "base_positive_words": self._base_positive_words,
            "base_negative_words": self._base_negative_words,
            "negations": self._negations,
        }
        with open(fileout, 'a') as f:
            json.dump(model, f)

    def load(self, filein):
        pass

    def summary(self)-> str:
        print(str(self) + ':' +
                '\n      positive words     : ' + str(len(self._base_positive_words)) +
                '\n      negative words     : ' + str(len(self._base_negative_words)) +
                '\n      negation words     : ' + str(len(self._negations)) +
                '\n      weigth_added_words : ' + str(self._weigth_added_words) +
                '\n      use_frequency      : ' + str(self._use_frequency) +
                '\n      verbose            : ' + str(self._verbose) +
                '\n      tokenizer          : ' + str(self._tokenizer_name))

    def __str__(self) -> str:
        return "NaiveSA"

class NaiveSAPipeline(SentimentAnalysisPipeline):
    def __init__(self, **params):
        super().__init__(**params)
        self._params = params
        self._pipeline.steps.append(('naive_sa', NaiveSA()))
        self._pipeline.set_params(**params)
    
    def get_params(self, deep=None):
        return self._pipeline.get_params(deep)

    def set_params(self, **params):
        self._pipeline.set_params(**params)
        return self

    def save(self, fileout):
        if fileout is None: return
        fileout_name = '{}_{}.json'.format(fileout, str(self))
        with open(fileout_name, 'a') as f: f.write("[")
        first = True
        for p in self._pipeline:
            if not first:
                with open(fileout_name, 'a') as f: f.write(",")
            else: first = False
            p.save(fileout_name)
        with open(fileout_name, 'a') as f: f.write("]")

    def __str__(self) -> str:
        return "NaiveSAPipeline"