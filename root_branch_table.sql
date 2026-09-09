CREATE TABLE IF NOT EXISTS root_branch_table (
  序号 INT,
  级别 INT,
  编号1 INT,
  编号2 INT,
  编号3 INT,
  编号4 INT,
  类别 VARCHAR(20),
  x DOUBLE,
  y DOUBLE,
  z DOUBLE,
  粗细 DOUBLE,
  成熟度 DOUBLE,
  上节点 INT,
  下节点 INT,
  PRIMARY KEY (序号)
);

INSERT INTO root_branch_table VALUES (0, 1, 0, 0, NULL, NULL, '首节点', -1.1083, 0.1795, -7.4254, 0.1308, 1.0, NULL, NULL);
INSERT INTO root_branch_table VALUES (1, 1, 1, 0, NULL, NULL, '首节点', 1.014, 2.3398, -10.2321, 0.1411, 1.0, NULL, NULL);
INSERT INTO root_branch_table VALUES (2, 1, 2, 0, NULL, NULL, '首节点', -1.5353, 0.5676, -7.6926, 0.133, 1.0, NULL, NULL);
INSERT INTO root_branch_table VALUES (3, 1, 3, 0, NULL, NULL, '首节点', 0.8171, 1.4642, -11.7115, 0.1448, 1.0, NULL, NULL);
INSERT INTO root_branch_table VALUES (4, 1, 4, 0, NULL, NULL, '首节点', 0.5719, 1.2742, -11.375, 0.1429, 1.0, NULL, NULL);
INSERT INTO root_branch_table VALUES (5, 1, 5, 0, NULL, NULL, '首节点', -1.7634, 0.9606, -8.4309, 0.1336, 1.0, NULL, NULL);
INSERT INTO root_branch_table VALUES (6, 1, 6, 0, NULL, NULL, '首节点', -2.3317, -1.6681, -12.3317, 0.1414, 1.0, NULL, NULL);
INSERT INTO root_branch_table VALUES (7, 1, 7, 0, NULL, NULL, '首节点', -2.2009, -0.6468, -11.1352, 0.1407, 1.0, NULL, NULL);