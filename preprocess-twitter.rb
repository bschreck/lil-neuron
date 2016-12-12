# Ruby 2.0
# Reads stdin: ruby -n preprocess-twitter.rb
#
# Script for preprocessing tweets by Romain Paulus
# with small modifications by Jeffrey Pennington

def tokenize input
    input = input
        .gsub(/[-+]?[.\d]*[\d]+[:,.\d]*/, "<number> ")
        .gsub(/([!?.]){2,}/){ # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
            "#{$~[1]} <repeat> "
        }
        .gsub(/\b(\S*?)(.)\2{2,}\b/){ # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
            # TODO: determine if the end letter should be repeated once or twice (use lexicon/dict)
            $~[1] + $~[2] + " <elong> "
        }
        .gsub(/([^a-z0-9()<>'`\-]){2,}/){ |word|
            "#{word.downcase} <allcaps> "
        }

    return input
end

puts tokenize($_)
