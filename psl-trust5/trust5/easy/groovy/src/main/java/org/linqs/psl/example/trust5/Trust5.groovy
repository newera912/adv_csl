package org.linqs.psl.example.trust5;

import org.linqs.psl.application.inference.LazyMPEInference;
import org.linqs.psl.application.inference.MPEInference;
import org.linqs.psl.config.ConfigBundle;
import org.linqs.psl.config.ConfigManager;
import org.linqs.psl.database.Database;
import org.linqs.psl.database.DatabasePopulator;
import org.linqs.psl.database.DataStore;
import org.linqs.psl.database.Partition;
import org.linqs.psl.database.Queries;
import org.linqs.psl.database.ReadOnlyDatabase;
import org.linqs.psl.database.loading.Inserter;
import org.linqs.psl.database.rdbms.driver.H2DatabaseDriver;
import org.linqs.psl.database.rdbms.driver.H2DatabaseDriver.Type;
import org.linqs.psl.database.rdbms.RDBMSDataStore;
import org.linqs.psl.groovy.PSLModel;
import org.linqs.psl.model.atom.Atom;
import org.linqs.psl.model.predicate.StandardPredicate;
import org.linqs.psl.model.term.ConstantType;
import org.linqs.psl.utils.dataloading.InserterUtils;
import org.linqs.psl.utils.evaluation.printing.AtomPrintStream;
import org.linqs.psl.utils.evaluation.printing.DefaultAtomPrintStream;
import org.linqs.psl.utils.evaluation.statistics.ContinuousPredictionComparator;
import org.linqs.psl.utils.evaluation.statistics.DiscretePredictionComparator;
import org.linqs.psl.utils.evaluation.statistics.DiscretePredictionStatistics;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import groovy.time.TimeCategory;
import java.nio.file.Paths;

/**
 * A simple EasyLP example.
 * In this example, we try to determine if two people know each other.
 * The model uses two features: where the people lived and what they like.
 * The model also has options to include symmetry and transitivity rules.
 *
 * @author Jay Pujara <jay@cs.umd.edu>
 */
public class Trust5 {
	private static final String PARTITION_OBSERVATIONS = "observations";
	private static final String PARTITION_TARGETS = "targets";
	private static final String PARTITION_TRUTH = "truth";

	private Logger log;
	private DataStore ds;
	private PSLConfig config;
	private PSLModel model;

	/**
	 * Class for config variables
	 */
	private class PSLConfig {
		public ConfigBundle cb;

		public String experimentName;
		public String dbPath;
		public String dataPath;
		public String outputPath;

		public boolean sqPotentials = true;
		public Map weightMap = [
            "Adjacent":100,
            "T":100,
            "Prior":50
		];
		public boolean useTransitivityRule = false;
		public boolean useSymmetryRule = false;

		public PSLConfig(ConfigBundle cb) {
			this.cb = cb;

			this.experimentName = cb.getString('experiment.name', 'default');
			this.dbPath = cb.getString('experiment.dbpath', '/network/rit/lab/ceashpc/adil/psl_tmp5');
			this.dataPath = cb.getString('experiment.data.path', '../data');
			this.outputPath = cb.getString('experiment.output.outputdir', Paths.get('output', this.experimentName).toString());
            
			this.weightMap["Adjacent"] = cb.getInteger('model.weights.adjacent', weightMap["Adjacent"]);
			this.weightMap["T"] = cb.getInteger('model.weights.t', weightMap["T"]);
			this.weightMap["Prior"] = cb.getInteger('model.weights.prior', weightMap["Prior"]);

            /*
			this.weightMap["Lived"] = cb.getInteger('model.weights.lived', weightMap["Lived"]);
			this.weightMap["Likes"] = cb.getInteger('model.weights.likes', weightMap["Likes"]);
            
			this.weightMap["Transitivity"] = cb.getInteger('model.weights.transitivity', weightMap["Transitivity"]);
			this.weightMap["Symmetry"] = cb.getInteger('model.weights.symmetry', weightMap["Symmetry"]);
			this.weightMap["Prior"] = cb.getInteger('model.weights.prior', weightMap["Prior"]);
			this.useTransitivityRule = cb.getBoolean('model.rule.transitivity', false);
			this.useSymmetryRule = cb.getBoolean('model.rule.symmetry', false);
            */
		}
	}

	public Trust5(ConfigBundle cb) {
		log = LoggerFactory.getLogger(this.class);
		config = new PSLConfig(cb);
		ds = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, Paths.get(config.dbPath, 'Trust5').toString(), true), cb);
		//ds = new RDBMSDataStore(new PostgreSQLDriver("psl", true), config);
		model = new PSLModel(this, ds);
	}

	/**
	 * Defines the logical predicates used in this model
	 */
	private void definePredicates() {
		model.add predicate: "Adjacent", types: [ConstantType.UniqueID, ConstantType.UniqueID];
		model.add predicate: "T", types: [ConstantType.UniqueID, ConstantType.UniqueID];
	}

	/**
	 * Defines the rules for this model, optionally including transitivty and
	 * symmetry based on the PSLConfig options specified
	 */
	private void defineRules() {
		log.info("Defining model rules");
        
		//model.add(
		//	rule:  Adjacent(P1,P2) >> Adjacent(P2,P1),
		//	squared: config.sqPotentials,
		//	weight : config.weightMap["Adjacent"]
		//);

        model.add(
		    rule: ( Adjacent(P1,P2) & T(P1,P2) & Adjacent(P2,P3) & T(P2,P3)&Adjacent(P1,P2)& (P1-P3)&(P1-P2)&(P2-P3)) >> T(P1,P3),
			//rule:  (T(P1,P2) & T(P2,P3) & (P1-P3)&(P1-P2)&(P2-P3))  >> T(P1,P3),
			squared: config.sqPotentials,
			weight : config.weightMap["T"]
		);

        model.add(
            rule: ~T(P1,P2),
            squared:config.sqPotentials,
            weight: config.weightMap["Prior"]

        );



		log.debug("model: {}", model);
	}

	/**
	 * Load data from text files into the DataStore. Three partitions are defined
	 * and populated: observations, targets, and truth.
	 * Observations contains evidence that we treat as background knowledge and
	 * use to condition our inferences
	 * Targets contains the inference targets - the unknown variables we wish to infer
	 * Truth contains the true values of the inference variables and will be used
	 * to evaluate the model's performance
	 */
	private void loadData(Partition obsPartition, Partition targetsPartition, Partition truthPartition) {
		log.info("Loading data into database");

		Inserter inserter = ds.getInserter(Adjacent, obsPartition);
		InserterUtils.loadDelimitedData(inserter, Paths.get(config.dataPath, "adjacent_obs.txt").toString());

		inserter = ds.getInserter(T, obsPartition);
		InserterUtils.loadDelimitedData(inserter, Paths.get(config.dataPath, "T_obs.txt").toString());

		inserter = ds.getInserter(T, targetsPartition);
		InserterUtils.loadDelimitedData(inserter, Paths.get(config.dataPath, "T_targets.txt").toString());

		inserter = ds.getInserter(T, truthPartition);
		InserterUtils.loadDelimitedDataTruth(inserter, Paths.get(config.dataPath, "T_truth.txt").toString());
	}

	/**
	 * Run inference to infer the unknown Knows relationships between people.
	 */
	private void runInference(Partition obsPartition, Partition targetsPartition) {
		log.info("Starting inference");

		Date infStart = new Date();
		HashSet closed = new HashSet<StandardPredicate>([Adjacent]);
		Database inferDB = ds.getDatabase(targetsPartition, closed, obsPartition);
		LazyMPEInference mpe = new LazyMPEInference(model, inferDB, config.cb);
		mpe.mpeInference();
		mpe.close();
		inferDB.close();

		log.info("Finished inference in {}", TimeCategory.minus(new Date(), infStart));
	}

	/**
	 * Writes the output of the model into a file
	 */
	private void writeOutput(Partition targetsPartition) {
		Database resultsDB = ds.getDatabase(targetsPartition);
		PrintStream ps = new PrintStream(new File(Paths.get(config.outputPath, "trust_infer.txt").toString()));
		AtomPrintStream aps = new DefaultAtomPrintStream(ps);
		Set atomSet = Queries.getAllAtoms(resultsDB,T);
		for (Atom a : atomSet) {
			aps.printAtom(a);
		}

		aps.close();
		ps.close();
		resultsDB.close();
	}

	/**
	 * Run statistical evaluation scripts to determine the quality of the inferences
	 * relative to the defined truth.
	 */
	private void evalResults(Partition targetsPartition, Partition truthPartition) {
		Database resultsDB = ds.getDatabase(targetsPartition, [T] as Set);
		Database truthDB = ds.getDatabase(truthPartition, [T] as Set);
		DiscretePredictionComparator dpc = new DiscretePredictionComparator(resultsDB);
		ContinuousPredictionComparator cpc = new ContinuousPredictionComparator(resultsDB);
		dpc.setBaseline(truthDB);
		//	 dpc.setThreshold(0.99);
		cpc.setBaseline(truthDB);
		DiscretePredictionStatistics stats = dpc.compare(T);
		double mse = cpc.compare(T);
		log.info("MSE: {}", mse);
		log.info("Accuracy {}, Error {}",stats.getAccuracy(), stats.getError());
		log.info(
				"Positive Class: precision {}, recall {}",
				stats.getPrecision(DiscretePredictionStatistics.BinaryClass.POSITIVE),
				stats.getRecall(DiscretePredictionStatistics.BinaryClass.POSITIVE));
		log.info("Negative Class Stats: precision {}, recall {}",
				stats.getPrecision(DiscretePredictionStatistics.BinaryClass.NEGATIVE),
				stats.getRecall(DiscretePredictionStatistics.BinaryClass.NEGATIVE));

		resultsDB.close();
		truthDB.close();
	}

	public void run() {
		log.info("Running experiment {}", config.experimentName);

		Partition obsPartition = ds.getPartition(PARTITION_OBSERVATIONS);
		Partition targetsPartition = ds.getPartition(PARTITION_TARGETS);
		Partition truthPartition = ds.getPartition(PARTITION_TRUTH);

		definePredicates();
		defineRules();
		loadData(obsPartition, targetsPartition, truthPartition);
		runInference(obsPartition, targetsPartition);
		writeOutput(targetsPartition);
        println 'Output Generated.'
		evalResults(targetsPartition, truthPartition);

		ds.close();
	}

	/**
	 * Parse the command line options and populate them into a ConfigBundle
	 * Currently the only argument supported is the path to the data directory
	 * @param args - the command line arguments provided during the invocation
	 * @return - a ConfigBundle populated with options from the command line options
	 */
	public static ConfigBundle populateConfigBundle(String[] args) {
		ConfigBundle cb = ConfigManager.getManager().getBundle("Trust5");
		if (args.length > 0) {
			cb.setProperty('experiment.data.path', args[0]);
		}
		return cb;
	}

	/**
	 * Run this model from the command line
	 * @param args - the command line arguments
	 */
	public static void main(String[] args) {
		ConfigBundle configBundle = populateConfigBundle(args);
        Trust5 lp = new Trust5(configBundle);
		lp.run();
        println "It is running now"
	}
}
