
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.logging.RedwoodConfiguration;


/** This class demonstrates building and using a Stanford CoreNLP pipeline. */
public class NLPSpanish {
// 
  private NLPSpanish() { } // static meain metho
  
  static StanfordCoreNLP pipeline;

	public static void main(String[] args) throws IOException {

		initialize();
		String[] filenames = {"pcori_cg_spanish_comments"
		};
		for(int i=0;i<filenames.length;i++) {
			try (BufferedReader br = new BufferedReader(new FileReader(filenames[i] + ".txt"))) {
				@SuppressWarnings("unused")
				StringBuilder str = new StringBuilder();
				int sum  = 0;
				int count = 0;
				for (String line; (line = br.readLine()) != null;) {
					Annotation annotation;
					annotation = new Annotation(line);
					pipeline.annotate(annotation);
					List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
					Sentiment senti = getSentiment(sentences.get(0));
					str.append(line.replace(',',';') + "," + senti.getSentimentName() + "," + senti.getSentimentScore()).append("\n");
					sum += senti.getSentimentScore();
					count++;
				}
				try (BufferedWriter br1 = new BufferedWriter(new FileWriter(filenames[i] + ".csv"))) {
					br1.write(str.toString());
				}
				System.out.println("The total count of " + filenames[i] + " = " + count);
				System.out.println("The total sum of " + filenames[i] + " = " + sum);
				System.out.println("The average score of " + filenames[i] + " = " + ((double)sum/(double)count));
			}
		}
	}

	private static void initialize() throws IOException  {
		Properties props = new Properties();
		props.load(IOUtils.readerFromString("StanfordCoreNLP-spanish.properties"));
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