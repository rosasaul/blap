#!/usr/bin/perl

use Math::Trig;

my $max_weight = 0.5;
my $max_iterations = 3000;
my $stop_threshold = 0.0001;

my $learning_rate = 0.1;
my $momentum = 0.05;

my $momentum_match_original = 1.2;
my $momentum_match_w1 = 1.2;
my $momentum_match_w2 = 1.2;
my $momentum_max = 2.5;

my $momentum_step = 0.1;

my $momentum_reverse = 0.5;

my %data;
readData('TEST',"data/iris_test.dat");
readData('TRAIN',"data/iris_training.dat");
readData('VALIDATE',"data/iris_validation.dat");

print "epoch,train,test,validation\n";

# initialize weight matrix
my $w1 = initMatrix(5,5);
my $weight_delta_w1_past = initMatrix(5,5);
#pMatrix($w1);
my $bias1 = 1;
my $w2 = initMatrix(3,6);
my $weight_delta_w2_past = initMatrix(3,6);
#pMatrix($w2);
my $bias2 = 1;

my $epoch_count = 0;

my $prev_error = 1;
my $train_count = 1 * @{ $data{'TRAIN'} };

while($epoch_count < $max_iterations){
  # Select random training row
  my $sample_index = int(rand($train_count));

  my @train_vector;
  foreach my $val (sort keys %{$data{'TRAIN'}[$sample_index]{'data'}}){
    push @train_vector, $data{'TRAIN'}[$sample_index]{'data'}{$val};
  }
  push @train_vector, $bias1;
#  print STDERR "first train_vector\n";
#  pVector(\@train_vector);

  # feed forward
  #  net = train_item * weights
  my @mid_vec = mulVecMat(\@train_vector, $w1);
#  print STDERR "mid_vec\n";
#  pVector(\@mid_vec);

  my @mid_vec_derivative;
  for(my $i = 0; $i < 5; $i++){
    $mid_vec_derivative[$i] = sigmoid_derivative($mid_vec[$i]);
    $mid_vec[$i] = sigmoid($mid_vec[$i]);
  }
  push @mid_vec, $bias2;
#  print STDERR "mid_vec after sigmoid\n";
#  pVector(\@mid_vec);
#  print STDERR "mid_vec_derivative\n";
#  pVector(\@mid_vec_derivative);

  my @out_vec = mulVecMat(\@mid_vec, $w2);
  my @out_vec_derivative;
  for(my $i = 0; $i < 3; $i++){
    $out_vec_derivative[$i] = sigmoid_derivative($out_vec[$i]);
    $out_vec[$i] = sigmoid($out_vec[$i]);
  }
#  pVector(\@out_vec);

  # Back propegation
  for(my $i = 0; $i < 3; $i++){
    $errorW2[$i] = $out_vec[$i] - $data{'TRAIN'}[$sample_index]{'label'}{$i};
  }
#  pVector(\@errorW2);

  my @delta_w2;
  for(my $i = 0; $i < 3; $i++){
    $delta_w2[$i] = $errorW2[$i] * $out_vec_derivative[$i];
  }

  my $weight_delta_w2 = initMatrixZero(3,6);
#  pMatrix($weight_delta_w1);
  for(my $i = 0; $i < 3; $i++){
    for(my $j = 0; $j < 6; $j++){
      $weight_delta_w2->[$i][$j] = -$learning_rate * $mid_vec[$j] * $delta_w2[$i];
    }
  }
#  pMatrix($weight_delta_w2);

  # Add Weight Delta to W2 layer
#  pMatrix($w2);
  for(my $i = 0; $i < 3; $i++){
    for(my $j = 0; $j < 6; $j++){
      my $momentum_add = 0;
      if(
        $weight_delta_w2->[$i][$j] > 0 and $weight_delta_w2_past->[$i][$j] > 0 or
        $weight_delta_w2->[$i][$j] < 0 and $weight_delta_w2_past->[$i][$j] < 0
      ){
        $momentum_add = $momentum_match * $weight_delta_w2_past->[$i][$j];
        if($momentum_match_w2 < $momentum_max){ $momentum_match_w2 += $momentum_step; }
      }
      else{
        $momentum_match_w2 = $momentum_match_original;
        $momentum_add = $momentum_reverse * $weight_delta_w2_past->[$i][$j];
      }

#      $w2->[$i][$j] += $weight_delta_w2->[$i][$j] + $momentum * $weight_delta_w2_past->[$i][$j];
      $w2->[$i][$j] += $weight_delta_w2->[$i][$j] + $momentum_add;
    }
  }
#  pMatrix($w2);

  ### Backprop top layer W1 ###

  # Error to pass to W1 from W2
  my @errorW1;
  for(my $i = 0; $i < 3; $i++){
    for(my $j = 0; $j < 6; $j++){
      $errorW1[$j] += $errorW2[$i] * $w2->[$i][$j];
    }
  }
#  pVector(\@errorW1);
  
  my $bias_delta = pop @errorW1;
#  $bias2 += -$learning_rate * $bias_delta * sigmoid_derivative($bias2);

  my @delta_w1;
  for(my $i = 0; $i < 5; $i++){
    $delta_w1[$i] = $errorW1[$i] * $mid_vec_derivative[$i];
  }

  # Weights delta previous layer
  my $weight_delta_w1 = initMatrixZero(5,5);
#  pMatrix($weight_delta_w1);
  for(my $i = 0; $i < 5; $i++){
    for(my $j = 0; $j < 5; $j++){
      $weight_delta_w1->[$i][$j] = -$learning_rate * $train_vector[$j] * $delta_w1[$i];
    }
  }
#  pMatrix($weight_delta_w1);

  # Add Weight Delta to W1 layer
#  pMatrix($w1);
  for(my $i = 0; $i < 5; $i++){
    for(my $j = 0; $j < 5; $j++){
      my $momentum_add = 0;
      if(
        $weight_delta_w1->[$i][$j] > 0 and $weight_delta_w1_past->[$i][$j] > 0 or
        $weight_delta_w1->[$i][$j] < 0 and $weight_delta_w1_past->[$i][$j] < 0
      ){
        $momentum_add = $momentum_match_w1 * $weight_delta_w1_past->[$i][$j];
        if($momentum_match_w1 < $momentum_max){ $momentum_match_w1 += $momentum_step; }
      }
      else{
        $momentum_match_w1 = $momentum_match_original;
        $momentum_add = $momentum_reverse * $weight_delta_w1_past->[$i][$j];
      }

      $w1->[$i][$j] += $weight_delta_w1->[$i][$j] + $momentum_add;

#      $w1->[$i][$j] += $weight_delta_w1->[$i][$j] + $momentum * $weight_delta_w1_past->[$i][$j];;
    }
  }
#  pMatrix($w1);
  
  $epoch_count++;

  my $accuracyTest = check_error('TEST',$w1,$bias1,$w2,$bias2);
  my $accuracyTrain = check_error('TRAIN',$w1,$bias1,$w2,$bias2);
  my $accuracyValidate = check_error('VALIDATE',$w1,$bias1,$w2,$bias2);

  print "$epoch_count,$accuracyTrain,$accuracyTest,$accuracyValidate\n";

  #print STDERR "epoch $epoch_count accuracy ".sprintf("%.2f",$accuracy * 100)."%\n";
  if($accuracyValidate eq 1){ last; }
  #if($prev_error - $error < $stop_threshold){ last; }
  #$prev_error = $error;

  for(my $i = 0; $i < 3; $i++){
    for(my $j = 0; $j < 6; $j++){
      $weight_delta_w2_past->[$i][$j] = $weight_delta_w2->[$i][$j];
    }
  }
  for(my $i = 0; $i < 5; $i++){
    for(my $j = 0; $j < 5; $j++){
      $weight_delta_w1_past->[$i][$j] = $weight_delta_w1->[$i][$j];
    }
  }
}

exit;

####### SUBS #######
sub check_error{
  my ($mode,$w1,$bias1,$w2,$bias2) = @_;
  my $total = 0; my $correct = 0;
  foreach my $item (@{ $data{$mode} }){
    my @vector;
    foreach my $val (sort keys %{$item->{'data'}}){
      push @vector, $item->{'data'}{$val};
    }
    push @vector, $bias1;
    #pVector(\@vector); die;

    my @mid_vec = mulVecMat(\@vector, $w1);
    for(my $i = 0; $i < @mid_vec; $i++){
      $mid_vec[$i] = sigmoid($mid_vec[$i]);
    }
    push @mid_vec, $bias2;
    #pVector(\@mid_vec); die;
    my @out_vec = mulVecMat(\@mid_vec, $w2);
    for(my $i = 0; $i < @out_vec; $i++){
      $out_vec[$i] = sigmoid($out_vec[$i]);
    }
    #pVector(\@out_vec); die;

#    if($item->{'label'}{0}){ print STDERR "label 0"; }
#    elsif($item->{'label'}{1}){ print STDERR "label 1"; }
#    elsif($item->{'label'}{2}){ print STDERR "label 2"; }

    if($item->{'label'}{0} eq 1 and $out_vec[0] > $out_vec[1] and $out_vec[0] > $out_vec[2]){ $correct++; }
    elsif($item->{'label'}{1} eq 1 and $out_vec[1] > $out_vec[0] and $out_vec[1] > $out_vec[2]){ $correct++; }
    elsif($item->{'label'}{2} eq 1 and $out_vec[2] > $out_vec[1] and $out_vec[2] > $out_vec[0]){ $correct++; }
    $total++;
  }
  return $correct / $total;
}
sub pMatrix{
  my $matrix = $_[0];
  my $output = "::Matrix::\n";
  for(my $i = 0; $i < @{$matrix}; $i++){
    $output .= "|";
    for(my $j = 0; $j < @{$matrix->[$i]}; $j++){
      $output .= " ".$matrix->[$i][$j];
    }
    $output .= " |\n";
  }
  $output .= "::Matrix::\n\n";
  print STDERR $output;
}
sub pVector{
  my $vector = $_[0];
  my $output = "::Vector::\n[";
  for(my $i = 0; $i < @{$vector}; $i++){
    $output .= " ".$vector->[$i];
  }
  $output .= "]\n::Vector::\n\n";
  print STDERR $output;
}
sub sigmoid{
  my $x = $_[0];
  return ((tanh($x) + 1) / 2);
}
sub sigmoid_derivative{
  my $x = $_[0];
  return ((1 - (tanh($x) * tanh($x)))/2);
}
sub mulVecMat{
  my $vector = $_[0];
  my $matrix = $_[1];
  my @output;
  for(my $i = 0; $i < @{$matrix}; $i++){

    for(my $j = 0; $j < @{$vector}; $j++){
      $output[$i] += $vector->[$j] * $matrix->[$i][$j];
    }

  }
  return @output;
}
sub initMatrix{
  my $rows = $_[0];
  my $cols = $_[1];
  my @matrix;
  for(my $i = 0; $i < $rows; $i++){
    for(my $j = 0; $j < $cols; $j++){
      $matrix[$i][$j] = $max_weight * rand();
    }
  }
  return \@matrix;
}
sub initMatrixZero{
  my $rows = $_[0];
  my $cols = $_[1];
  my @matrix;
  for(my $i = 0; $i < $rows; $i++){
    for(my $j = 0; $j < $cols; $j++){
      $matrix[$i][$j] = 0;
    }
  }
  return \@matrix;
}

sub readData {
  my $mode = $_[0];
  my $file = $_[1];
  my @set;
  
  open IN, "<".$file or die "Failed to open $file : $!\n";
  while(my $line = <IN>){ chomp($line);
    #print STDERR "line : $line\n";
    my @items = split(/\s+/,$line);
    my %item;
    $item{'data'}{0} = $items[0];
    $item{'data'}{1} = $items[1];
    $item{'data'}{2} = $items[2];
    $item{'data'}{3} = $items[3];

    $item{'label'}{0} = $items[4];
    $item{'label'}{1} = $items[5];
    $item{'label'}{2} = $items[6];
    push @set, \%item;
  }
  close IN;
  $data{$mode} = \@set;
}
