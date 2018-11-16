
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.logging.RedwoodConfiguration;

public class NLP {

	static StanfordCoreNLP pipeline;

	public static void main(String[] args) throws IOException {

		initialize();
		try (BufferedReader br = new BufferedReader(new FileReader("pcori_patient_english.txt"))) {
			for (String line; (line = br.readLine()) != null;) {
				Annotation annotation;
				annotation = new Annotation(line);
				pipeline.annotate(annotation);
				List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
				Sentiment senti = getSentiment(sentences.get(0));
				System.out.println(line.replace(',',';') + "," + senti.getSentimentName() + "," + senti.getSentimentScore());
			}
		}
	}

	private static void initialize() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref, sentiment");
		RedwoodConfiguration.current().clear().apply();
		pipeline = new StanfordCoreNLP(props);

	}

	public static Sentiment getSentiment(CoreMap sentence) {
		String name = sentence.get(SentimentCoreAnnotations.SentimentClass.class);
		Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
		int sentiment = RNNCoreAnnotations.getPredictedClass(tree);
		Sentiment senti = new Sentiment();
		senti.setSentimentName(name);
		senti.setSentimentScore(sentiment);
		return senti;
	}

}