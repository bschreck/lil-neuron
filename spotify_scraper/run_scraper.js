require('coffee-script').register()
var fs = require('fs');
var path = require('path');
var program = require('commander');
var prompt = require('prompt');

program
  .command('startlocal')
  .description('Start server locally')
  .action(startLocal)

program.version(require('./package.json').version);

program.parse(process.argv);
// Default command ?
if (program.args.length === 0) {
    start();
    process.exit();
}
function startLocal() {
  var env = "local";
  start(env);
}

function start(env) {
    var express = require('express');
    var app = express();

    var bodyParser = require('body-parser');

    app.use(bodyParser.json({limit: '50mb'}));
    app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));

    if (env == "local") {
      var port = process.env.PORT || 8080;
    } else {
      var port = process.env.PORT || 80;
    }
    console.log(port);

    /*
     *var router = express.Router()
     *app.use('/api', router);
     */

    var mongoose = require('mongoose');
    mongoose.connect("mongodb://localhost:27017/lil-neuron-db");

    /*
     *require('./app/routes')(router, models, dbFunctions, utils);
     */

    /*
     *app.listen(port);
     */
    var Genre = require('./genre')
    var Artist = require('./artist')
    var scraper = require('./scraper');
    scraper.getArtists(Genre, Artist);
}
