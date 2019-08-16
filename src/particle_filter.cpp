#include "../include/particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

#include "../include/helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);
    std::default_random_engine gen;
    num_particles = 100;  // TODO: Set the number of particles
    for (int i = 0; i < num_particles; i++) {
        particles.emplace_back(Particle{i, dist_x(gen), dist_y(gen), dist_theta(gen), 1});
    }
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

    std::default_random_engine gen;

    // generate random Gaussian noise
    std::normal_distribution<double> N_x(0, std_pos[0]);
    std::normal_distribution<double> N_y(0, std_pos[1]);
    std::normal_distribution<double> N_theta(0, std_pos[2]);
   for (auto &particle : particles) {
       if (fabs(yaw_rate) < Util::Precision()) { // if velocity is constant
           particle.x += velocity * delta_t * cos(particle.theta);
           particle.y += velocity * delta_t * sin(particle.theta);
       } else {
           particle.x += velocity / yaw_rate * ( sin( particle.theta + yaw_rate*delta_t ) - sin(particle.theta) );
           particle.y += velocity / yaw_rate * ( cos( particle.theta ) - cos( particle.theta + yaw_rate*delta_t ) );
           particle.theta += yaw_rate * delta_t;
       }


       particle.x += N_x(gen);
       particle.y += N_y(gen);
       particle.theta += N_theta(gen);
   }


}

void ParticleFilter::dataAssociation(const vector<Util::LandmarkObs>& predicted,
                                     vector<Util::LandmarkObs>& observations) {
  /**
   *  Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   */
    for (auto &observed_landmark : observations) {
        observed_landmark.id = Util::NearestNeighbor(predicted, observed_landmark).id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<Util::LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   *  Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */


   for (auto &particle : particles) {
       particle.weight = 1;
       // transform the coordinates measured in vehicle space to particle space using homogenous transformation
       vector<Util::LandmarkObs> transformed_observation;
       for (auto &observation : observations) {
           transformed_observation.emplace_back(Util::HomogenousTransformation(observation, particle.x, particle.y, particle.theta));
       }

       // get the predicted landmarks based on sensor_range
       vector<Util::LandmarkObs> predicted_landmarks;
       std::unordered_map<int, Util::LandmarkObs> id_map;
       for (auto &landmakr : map_landmarks.landmark_list) {
           if (fabs(landmakr.x_f - particle.x) <= sensor_range && fabs(landmakr.y_f - particle.y) <= sensor_range) {
               predicted_landmarks.emplace_back(Util::LandmarkObs{landmakr.id_i, landmakr.x_f, landmakr.y_f});
               id_map[landmakr.id_i] = predicted_landmarks[predicted_landmarks.size() - 1];
           }
       }

       // Associate the each trasnformed observation to their cloest landmarks
       dataAssociation(predicted_landmarks, transformed_observation);

       // update weight using the multivariate gussian distribution
       for (auto &transformed_obs : transformed_observation) {
           particle.weight *= Util::MultiVariateGaussian(std_landmark, transformed_obs, id_map[transformed_obs.id]);
       }
   }

}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

    vector<Particle> new_particles;
    std::default_random_engine gen;
    // get all of the current weights
    vector<double> weights;
    double max_weight = 0;
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
        max_weight = std::max(max_weight, particles[i].weight);
    }

    // generate random starting index for resampling wheel
    std::uniform_int_distribution<int> uniintdist(0, num_particles-1);
    auto index = uniintdist(gen);

    // uniform random distribution [0.0, max_weight)
    std::uniform_real_distribution<double> unirealdist(0.0, max_weight);

    double beta = 0.0;

    // spin the resample wheel!
    for (int i = 0; i < num_particles; i++) {
        beta += unirealdist(gen) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }

    particles = new_particles;

}

//Particle sample_based_on_weight() {
//    total_
//}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}